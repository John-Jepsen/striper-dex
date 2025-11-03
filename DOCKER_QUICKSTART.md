# 🐳 Docker Quick Start

## Instant Setup

```bash
make build    # Build images (2-3 min)
make train    # Train model (2-5 min)
```

**That's it!** Model saved to `./models/fishing_model_with_tidal.joblib`

---

## Common Commands

```bash
make help              # Show all commands
make collect           # Collect all data
make pipeline          # Run full pipeline
make forecast          # Generate predictions
make shell             # Open bash in container
```

---

## What You Get

✅ **Containerized environment** - No Python setup needed  
✅ **Data persistence** - All data/models saved to host  
✅ **Multi-service** - Train, predict, visualize separately  
✅ **Production ready** - Consistent across environments

---

## Services

| Service | Command | Description |
|---------|---------|-------------|
| Train | `make train` | Train model (R²=0.72) |
| Collect | `make collect` | Gather all data |
| Features | `make features` | Engineer features |
| Forecast | `make forecast` | Predict fishing |
| Visualize | `make visualize` | Create plots |
| Compare | `make compare` | Compare models |

---

## Directory Mapping

```
Host Machine          →  Container
./data/              →  /app/data/
./models/            →  /app/models/
./figures/           →  /app/figures/
```

All files persist on your machine!

---

## Full Documentation

See [README.docker.md](README.docker.md) for complete guide.

---

## Troubleshooting

**Build fails?**
```bash
make clean
make build
```

**Need logs?**
```bash
make logs
```

**Want to explore?**
```bash
make shell
```
