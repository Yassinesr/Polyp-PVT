#!/usr/bin/env bash
# List checkpoints not modified in 30 days for manual review
find model_pth -maxdepth 1 -type d -mtime +30 -exec du -sh {} \; | sort -h
