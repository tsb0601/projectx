#!/bin/bash

export USER=ellisbrown
export TOKEN=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
export REPOSITORY=visionx-cambrian/llava_base
git clone https://${USER}:${TOKEN}@github.com/${REPOSITORY} ~/llava_base