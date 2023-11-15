import fs from "fs";
import matter from "gray-matter";
import md from "markdown-it";
import mdgh from "markdown-it-github-headings";
import Tags from "../../components/Tags";
import dateFormat from "dateformat";
import React, { useEffect } from 'react';

export default function Post({ frontmatter, content, slug }) {
  useEffect(() => {
    // Check if MathJax object is available
    if (window.MathJax) {
      // Typeset / reprocess the math content
      window.MathJax.typesetPromise();
    }
  }, [content]);
  return (
    <div className="w-full flex justify-center py-5 pt-16 md:pt-5">
      <Tags
        title={frontmatter.title}
        desc={md({ html: true }).use(mdgh, {prefixHeadingIds: false}).render(content).slice(0, 157) + "..."}
        image={frontmatter.previewImg}
        slug={"/blog/" + slug}
      />
      <div className="container px-5" lang="en">
        <h1
          lang="en"
          style={{ hyphens: "auto" }}
          className="text-4xl md:text-4xl w-full font-bold break-words"
        >
          {frontmatter.title}
        </h1>
        <p className="text-xl pt-2 pb-2">
          by: {frontmatter.author},{" "}
          {dateFormat(frontmatter.date, "mmm dd, yyyy")}
        </p>
        <hr />
        <div
          className="pt-2 article"
          dangerouslySetInnerHTML={{
            __html: md({ html: true }).use(mdgh, {prefixHeadingIds: false}).render(content),
          }}
        />
      </div>
    </div>
  );
}

export async function getStaticProps({ params: { slug } }) {
  const fileName = fs.readFileSync(`blog/${slug}.md`, "utf-8");
  const { data: frontmatter, content } = matter(fileName);
  return {
    props: {
      frontmatter,
      content,
      slug,
    },
  };
}

export async function getStaticPaths() {
  const files = fs.readdirSync("blog");

  const paths = files.map((fileName) => ({
    params: {
      slug: fileName.replace(".md", ""),
    },
  }));

  return {
    paths,
    fallback: false,
  };
}
