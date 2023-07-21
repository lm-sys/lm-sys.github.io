import { readdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import RSS from "rss";

const rootUrl = "https://lmsys.org";
const projectRoot = path.join(fileURLToPath(import.meta.url), "..", "..");

/**
 * Loads contents of blog posts
 */
const loadBlogPosts = async () => {
  const blogDir = path.join(projectRoot, "blog");
  const blogPostPaths = await readdir(blogDir);

  return (
    await Promise.all(
      blogPostPaths.map(async (filePath) => [
        filePath,
        (await readFile(path.join(projectRoot, "blog", filePath))).toString(),
      ])
    )
  ).reduce(
    (agg, [fileName, fileContents]) => ({ [fileName]: fileContents, ...agg }),
    {}
  );
};

/**
 * Given a blog metadata field, strips label & cleans
 */
const cleanBlogMetadataField = (input) =>
  input
    .split(":")
    .slice(1)
    .join(":")
    .trim()
    .replaceAll(`"`, "")
    .replaceAll(`\\`, "");

/**
 * Given a blog post head, parses out title, author, date & preview
 */
const parseBlogPostHeader = (content) => {
  const [title, authors, date, url] = content
    .split("\n")
    .slice(1, 5)
    .map(cleanBlogMetadataField);

  return {
    title,
    authors,
    date: new Date(date),
    url,
  };
};

const genRssFeed = async () => {
  const rss = new RSS({
    title: "Large Model Systems Organization",
    description: `Large Model Systems Organization (LMSYS Org) is an open research organization founded by students and faculty from UC Berkeley in collaboration with UCSD and CMU. We aim to make large models accessible to everyone by co-development of open models, datasets, systems, and evaluation tools. Our work encompasses research in both machine learning and systems. We train large language models and make them widely available, while also developing distributed systems to accelerate their training and inference`,
    site_url: rootUrl,
    image_url: `${rootUrl}/public/images/gallery/universe.png`,
  });

  const posts = await loadBlogPosts();
  Object.keys(posts).forEach((postFilename) => {
    const { title, authors, date } = parseBlogPostHeader(posts[postFilename]);

    rss.item({
      title,
      // rm ending '.md'
      url: `${rootUrl}/blog/${postFilename.slice(0, postFilename.length - 3)}/`,
      author: authors,
      date,
    });
  });

  return rss.xml();
};

const writeRssFeed = async () => {
  const xml = await genRssFeed();
  await writeFile(path.join(projectRoot, "public", "rss.xml"), xml);
};

await writeRssFeed();
