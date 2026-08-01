using Documenter
using Reels

makedocs(
    modules=[Reels],
    sitename="Reels",
    remotes=nothing,
    checkdocs=:exports,
    format=Documenter.HTML(
        prettyurls=true,
        edit_link=nothing,
        repolink=nothing,
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "Configuration" => "configuration.md",
        "Data" => [
            "Manifest and cache" => "data/manifest_and_cache.md",
        ],
        "Training" => [
            "CLI" => "training/cli.md",
            "Distributed" => "training/distributed.md",
            "Low VRAM" => "training/low_vram.md",
        ],
        "Operations" => [
            "Resume and recovery" => "resume_and_recovery.md",
            "Troubleshooting" => "troubleshooting.md",
        ],
        "Licensing" => "licensing.md",
        "Public API" => "api.md",
    ],
)
