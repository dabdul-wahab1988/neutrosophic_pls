# PyPI Publishing Setup

This document explains how to set up Trusted Publishing for automated PyPI releases.

## Overview

The project uses GitHub Actions with **Trusted Publishing** (OpenID Connect) to automatically publish releases to PyPI. This is the modern, secure approach recommended by PyPI and eliminates the need to manage API tokens.

## Setting Up Trusted Publishing

Before the workflow can publish to PyPI, you need to configure Trusted Publishing on PyPI:

### 1. Go to PyPI Account Settings

Visit: https://pypi.org/manage/account/publishing/

### 2. Add a New Trusted Publisher

Configure the following details:
- **PyPI Project Name**: `neutrosophic-pls`
- **Owner**: `dabdul-wahab1988`
- **Repository name**: `neutrosophic_pls`
- **Workflow name**: `python-publish.yml`
- **Environment name**: (leave empty)

### 3. Save the Configuration

Once saved, GitHub Actions can publish to PyPI without needing API tokens.

## How It Works

1. When you create a new release on GitHub (with a tag like `v1.0.1`), the workflow triggers
2. The workflow builds the Python package
3. GitHub generates a short-lived OIDC token
4. PyPI verifies the token and allows the upload

## Creating a Release

To publish a new version:

1. Update the version in `pyproject.toml`
2. Commit and push the changes
3. Create a new release on GitHub with a tag (e.g., `v1.0.1`)
4. The workflow automatically builds and publishes to PyPI

## Troubleshooting

### First Release Setup

For the **first release**, PyPI may require you to:
1. Manually create the project on PyPI first, OR
2. Use the "Add a new pending publisher" option on PyPI before creating the release

### Common Issues

- **403 Error**: Ensure Trusted Publishing is configured correctly on PyPI
- **Project doesn't exist**: Create the project manually on PyPI first, or use pending publisher
- **Version already exists**: Increment the version number in `pyproject.toml`

## Benefits of Trusted Publishing

- ✅ No API tokens to manage or rotate
- ✅ More secure (uses short-lived tokens)
- ✅ No secrets stored in GitHub
- ✅ Recommended by PyPI
- ✅ Automatic revocation when repository is deleted

## References

- [PyPI Trusted Publishing Documentation](https://docs.pypi.org/trusted-publishers/)
- [Python Packaging Guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
