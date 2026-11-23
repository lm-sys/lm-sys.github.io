import { readdir, readFile, writeFile } from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";
import matter from "gray-matter";
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
 * Reads blog metadata by key so optional fields can appear in any order.
 */
const parseBlogPostHeader = (content) => {
  const { data } = matter(content);

  return {
    title: data.title,
    authors: data.author,
    date: new Date(data.date),
    externalUrl: data.externalUrl,
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
    const { title, authors, date, externalUrl } = parseBlogPostHeader(posts[postFilename]);

    rss.item({
      title,
      // rm ending '.md'
      url: externalUrl || `${rootUrl}/blog/${postFilename.slice(0, postFilename.length - 3)}/`,
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
