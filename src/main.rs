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
            let profile = profiler::profile_input(&source, threshold)?;
            let targets = analyzer::select_targets(&profile, threshold, ml_mode);
            let generation = generator::generate(&source, &targets, &output, dry_run)?;
            builder::build_extension(&generation, dry_run)?;

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
