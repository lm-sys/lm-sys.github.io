# Repository Notes

## Local Development

- Use the Node version in `.nvmrc` (`v20`) when possible.
- Install dependencies with `npm i`.
- The README development flow is:

```bash
npm i
npm run build
npm run dev
```

- `npm run dev` runs `npm run rss` first, then starts Next.js on `http://localhost:3000`.
- `npm run build` also regenerates `public/rss.xml` before building.
- `npm run deploy` builds, exports, optimizes images for static export, and touches `out/.nojekyll`.

## Content Layout

- Blog posts live in `blog/*.md`.
- A new blog post is picked up automatically by `src/pages/blog/[slug].js`; the URL slug is the markdown filename without `.md`.
- Blog post frontmatter should include `title`, `author`, `date`, and `previewImg`.
- Keep the first four frontmatter fields in this order when practical: `title`, `author`, `date`, `previewImg`. `scripts/gen-rss.mjs` parses those first four header lines directly for the RSS feed.
- Non-blog content is mostly under `content/`, for example about, FAQ, impressum, team, and Vicuna evaluation content.

## Images and Media

- Put site images under `public/images`.
- Put blog-specific images under `public/images/blog/<post-slug>/`.
- Reference public assets from markdown and JSX with root-relative paths, for example `/images/blog/routellm/cover.png`, not `public/images/...`.
- Blog card preview images use `frontmatter.previewImg` through `next-image-export-optimizer` in `src/pages/blog/index.js`.
- Blog article pages render markdown with `markdown-it` and `html: true`, so existing posts often use raw HTML such as `<img src="/images/blog/...">` for sizing and centering.
- Markdown image syntax also works in posts, for example `![caption](/images/blog/<post-slug>/figure.png)`.
- MathJax is loaded from `src/pages/_document.js` using `public/mathjax.js` plus the CDN script.

## Build Notes

- A successful build currently emits warnings for outdated Browserslist data.
- On Node 22, the build and dev server work, but Node prints `DEP0040` `punycode` deprecation warnings. Prefer Node 20 from `.nvmrc` for less noise.
- Static generation currently warns that highlight.js cannot find the `bibtex` language for posts using BibTeX code fences.
- Existing homepage lint warnings include a missing `useEffect` dependency in `src/pages/index.js` and an `<img>` without Next Image/alt handling.
