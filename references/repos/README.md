# External Repository References

This directory contains git submodules referencing external repositories used in this project.

## nagengast25-ma-thesis

**Source:** https://gitlab.uni-hannover.de/nagengast.m/nagengast25-ma-thesis.git

This submodule provides the NRC Lexicon datasets used for emotion/sentiment analysis. We use **sparse-checkout** to only materialize the `data/NRC-Lexicons` subdirectory, keeping the checkout minimal.

### Initial Setup (after cloning the main project)

```bash
# Initialize and clone all submodules
git submodule update --init --recursive

# Configure sparse-checkout for the thesis repo
cd references/repos/nagengast25-ma-thesis
git sparse-checkout init --cone
git sparse-checkout set data/NRC-Lexicons
```

### What's Included

After sparse-checkout, only these directories will be present:
- `data/NRC-Lexicons/NRC-Emotion-Intensity-Lexicon/`
- `data/NRC-Lexicons/NRC-Emotion-Lexicon/`
- `data/NRC-Lexicons/NRC-VAD-Lexicon-v2.1/`

### Updating the Submodule

To pull the latest changes from the remote:

```bash
cd references/repos/nagengast25-ma-thesis
git pull origin main
cd ../../..
git add references/repos/nagengast25-ma-thesis
git commit -m "Update nagengast25-ma-thesis submodule"
```

