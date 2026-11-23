# lmsys.org

This repo hosts all the content within https://lmsys.org. 

## Develop and Deploy
See the [package.json](./package.json)

1. Install `npm` and `node.js`


2. Develop:
```bash
npm i
npm run build
npm run dev
```

3. Deploy:

```bash
npm i
npm run build
npm run export
```

## Adding Blog Posts
Add a markdown file to `/blog` for it to be added in the blog feed. Make sure your frontmatter contains the entries `title`, `author`, `date`, and `previewImg`.

For an article hosted on another site, add `externalUrl` with its full HTTPS URL. The blog card and RSS entry will link directly to that article. Optionally set `description` for the card summary. Keep the original publication date and use a local `previewImg` as with other posts.

The Markdown body should contain a short summary and a link to the full article. It remains available at the LMSYS post URL as a fallback, with a canonical link to the external article. See `blog/2026-11-23-miles-true-on-policy.md` for an example.

## Editing Content
Some content is not sourced through Javascript, such as the about page, impressum, FAQ, and team list. For those, you can find their respective markdown and json files in `/content`

# Acknowledgement

The website is modified from the [laion.ai](https://github.com/LAION-AI/laion.ai.git), fully respecting [their license](https://github.com/LAION-AI/laion.ai/blob/main/LICENSE).
