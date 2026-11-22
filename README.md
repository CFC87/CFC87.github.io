# Fanchao Chen

Source for my personal homepage and blog, built by GitHub Pages with Jekyll.

## Structure

```
index.html        homepage (plain HTML, published as-is)
stylesheet.css    shared styles
blog/index.html   blog list page
blog.css          blog styles
_posts/           blog posts in Markdown
_layouts/         page templates for the blog
_includes/        small template pieces
images/           profile photo, favicon, icons, and logos
```

## Writing a post

Add a Markdown file to `_posts/` named `YYYY-MM-DD-title-in-kebab-case.md`:

```markdown
---
title: "Post Title"
summary: One or two sentences shown on the blog list.   # optional
cover: /images/blog/my-cover.jpg                        # optional
math: true                                              # optional, loads KaTeX
---

Post body in Markdown.
```

- The post is published at `/blog/title-in-kebab-case/`.
- Without `cover`, the list page draws a text cover from the title.
- Without `summary`, the first paragraph is used.
- Math uses `$$...$$`, both inline and on its own lines.

## Local preview

Requires Ruby 3.3 (for example `brew install ruby@3.3`).

```bash
export PATH=/opt/homebrew/opt/ruby@3.3/bin:$PATH
bundle config set --local path vendor/bundle
bundle install
bundle exec jekyll serve
```

Then open http://localhost:4000.

## Credits

Icons from [Font Awesome Free](https://fontawesome.com/license/free) (CC BY 4.0).
Company logos are trademarks of their respective owners.
