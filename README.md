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

## Editing Content
Some content is not sourced through Javascript, such as the about page, impressum, FAQ, and team list. For those, you can find their respective markdown and json files in `/content`

# Acknowledgement
The website is modified from the [laion.ai](https://github.com/LAION-AI/laion.ai.git), fully respecting [their license](https://github.com/LAION-AI/laion.ai/blob/main/LICENSE).
