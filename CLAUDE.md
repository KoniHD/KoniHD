# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **GitHub Pages profile site** built with:
- **Jekyll** - Static site generator
- **Academic Pages template** - Academic-focused theme
- **Ruby** language (via Bundler/Gems)
- **Node.js** for npm-based utilities

The site serves as a personal/academic portfolio at `https://kzeck.de`.

## Repository Structure

```
KoniHD/
├── _config.yml              # Main Jekyll configuration
├── _pages/                  # Static pages (about, cv, sitemap, etc.)
├── _posts/                  # Blog posts in chronological order
├── _portfolio/              # Portfolio/project items
├── _publications/           # Academic publications
├── _teaching/               # Teaching materials/courses
├── _data/                   # Site data (collections, metadata)
├── _sass/                   # Custom SCSS stylesheets
├── _includes/               # Reusable Jekyll components
├── _layouts/                # Page layouts
├── assets/                  # Static assets (CSS, JS, images)
└── markdown_generator/      # Ruby scripts for bulk content generation
```

## Key Configuration Files

### `_config.yml`
Main configuration with:
- Site metadata (title, description, email, social links)
- Theme settings (color schemes, fonts, icons)
- Navigation menu configuration
- Jekyll plugin list (jekyll-feed, jekyll-seo-tag, etc.)
- Disqus comment system integration

### `Gemfile` & `_config_docker.yml`
Ruby dependencies for Jekyll builds and Docker deployment.

## Common Commands

### Build the site locally
```bash
# Install Ruby gems
bundle install

# Build and serve (default port 4000)
bundle exec jekyll serve

# Build for production (generates _site/)
bundle exec jekyll build
```

### Docker deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Markdown generation scripts
The `markdown_generator/` directory contains Ruby scripts for bulk content creation:
- Generate publications from BibTeX
- Generate portfolio entries from CSV
- Automate talk/tutorial metadata

Run with: `bundle exec ruby <script_name>.rb [arguments]`

### NPM utilities
```bash
# See package.json for available Node.js scripts
npm run <script-name>
```

## Site Collections

Content is organized into Jekyll collections (subdirectories):
- `_pages/` - Static navigation pages
- `_posts/` - Blog posts (`YYYY-MM-DD-title.md` format)
- `_portfolio/` - Project/portfolio items
- `_publications/` - Academic papers/publications
- `_teaching/` - Teaching materials and courses

Each collection item follows a frontmatter-based configuration pattern with:
- `title`, `excerpt`, `category`, `tags`
- Date fields (`date`, `venue`, etc.)
- External links (`url`, `paper_url`, `project_url`)

## Development Workflow

1. Make changes to files in `_pages/`, `_posts/`, or other collections
2. Run `bundle exec jekyll serve` to preview locally
3. Commit changes - the site is version-controlled

The `_site/` directory is generated on build and should be `.gitignore`d.

## Verification Tags

Recent commits added SEO verification tags:
- Google Site Ownership
- Bing Site Ownership
- Yandex Verification

These are in appropriate locations for crawler detection.
