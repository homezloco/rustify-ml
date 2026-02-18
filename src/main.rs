use rustify_ml::analyzer;
use rustify_ml::builder;
use rustify_ml::generator;
use rustify_ml::input;
use rustify_ml::profiler;
use rustify_ml::utils;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing::{info, warn};

#[derive(Parser, Debug)]
#[command(
    name = "rustify-ml",
    about = "Accelerate Python ML hotspots with Rust stubs",
    version
)]
struct Args {
    #[command(subcommand)]
    command: Commands,
    /// Increase verbosity (-v, -vv)
    #[arg(short, long, action = clap::ArgAction::Count, global = true)]
    verbose: u8,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Profile and generate Rust bindings for Python code
    Accelerate {
        /// Path to a Python file to analyze
        #[arg(long)]
        file: Option<std::path::PathBuf>,
        /// Read Python code from stdin
        #[arg(long, default_value_t = false)]
        snippet: bool,
        /// Git repository URL to clone and analyze (optional)
        #[arg(long)]
        git: Option<String>,
        /// Path within the git repo to analyze (required when using --git)
        #[arg(long, value_name = "RELATIVE_PATH")]
        git_path: Option<std::path::PathBuf>,
        /// Minimum hotspot threshold percentage
        #[arg(long, default_value_t = 10.0)]
        threshold: f32,
        /// Output directory for generated extension
        #[arg(long, default_value = "dist")]
        output: std::path::PathBuf,
        /// Enable ML-focused heuristics
        #[arg(long, default_value_t = false)]
        ml_mode: bool,
        /// Print planned actions without executing
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// After building, run a Python timing harness and print speedup vs original
        #[arg(long, default_value_t = false)]
        benchmark: bool,
        /// Profile only: print hotspot table and exit without generating code
        #[arg(long, default_value_t = false)]
        list_targets: bool,
        /// Skip profiler and target a specific function by name
        #[arg(long)]
        function: Option<String>,
        /// Number of profiler loop iterations (default: 100)
        #[arg(long, default_value_t = 100u32)]
        iterations: u32,
    },
}

fn init_tracing(verbosity: u8) {
    let level = match verbosity {
        0 => "info",
        1 => "debug",
        _ => "trace",
    };

    let subscriber = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(level)
        .with_writer(std::io::stderr)
        .finish();

    let _ = tracing::subscriber::set_global_default(subscriber);
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing(args.verbose);

    match args.command {
        Commands::Accelerate {
            file,
            snippet,
            git,
            git_path,
            threshold,
            output,
            ml_mode,
            dry_run,
            benchmark,
            list_targets,
            function,
            iterations,
        } => {
            info!(
                ?file,
                snippet,
                ?git,
                ?git_path,
                threshold,
                ?output,
                ml_mode,
                dry_run,
                benchmark,
                list_targets,
                ?function,
                iterations,
                "starting accelerate"
            );
            let source = input::load_input(
                file.as_deref(),
                snippet,
                git.as_deref(),
                git_path.as_deref(),
            )?;
            let input_kind = match &source {
                utils::InputSource::File { path, .. } => format!("file:{}", path.display()),
                utils::InputSource::Snippet(_) => "snippet:stdin".to_string(),
                utils::InputSource::Git { repo, path, .. } => {
                    format!("git:{}:{}", repo, path.display())
                }
            };

            // --list-targets: profile only, print hotspot table, exit
            if list_targets {
                let profile = profiler::profile_input(&source, threshold)?;
                utils::print_hotspot_table(&profile.hotspots);
                info!(input_kind, "list-targets completed");
                return Ok(());
            }

            // Determine targets: --function bypasses profiler entirely
            let targets = if let Some(ref func_name) = function {
                info!(func = %func_name, "using --function: skipping profiler");
                vec![utils::TargetSpec {
                    func: func_name.clone(),
                    line: 1,
                    percent: 100.0,
                    reason: "--function flag".to_string(),
                }]
            } else {
                let profile = profiler::profile_input_with_iterations(&source, threshold, iterations)?;
                analyzer::select_targets(&profile, threshold, ml_mode)
            };

            let generation = if ml_mode {
                generator::generate_ml(&source, &targets, &output, dry_run)?
            } else {
                generator::generate(&source, &targets, &output, dry_run)?
            };
            builder::build_extension(&generation, dry_run)?;

            // Optional benchmark: run Python timing harness and print speedup
            if benchmark
                && !dry_run
                && let Err(e) = builder::run_benchmark(&source, &generation, &targets)
            {
                warn!(err = %e, "benchmark failed; skipping speedup output");
            }

            // Build summary rows from targets + generation result
            let summary_rows: Vec<utils::AccelerateRow> = targets
                .iter()
                .enumerate()
                .map(|(i, t)| {
                    let is_fallback = i < generation.generated_functions.len()
                        && generation.generated_functions[i].contains("// fallback");
                    utils::AccelerateRow {
                        func: t.func.clone(),
                        line: t.line,
                        pct_time: t.percent,
                        translation: if is_fallback { "Partial" } else { "Full" },
                        status: if is_fallback {
                            "Fallback: echo input".to_string()
                        } else {
                            "Success".to_string()
                        },
                    }
                })
                .collect();

            utils::print_summary(&summary_rows, &generation.crate_dir);

            if generation.fallback_functions > 0 {
                warn!(
                    input_kind,
                    fallback_functions = generation.fallback_functions,
                    "some targets used fallback translation; review generated code"
                );
            }
            info!(
                input_kind,
                targets = targets.len(),
                generated = generation.generated_functions.len(),
                fallback_functions = generation.fallback_functions,
                "accelerate flow completed"
            );
            if dry_run {
                info!("dry-run completed; no install performed");
            }
        }
    }

    Ok(())
}
