import fs from "fs";
import matter from "gray-matter";
import md from "markdown-it";
import Tags from "../components/Tags";
import mdgh from "markdown-it-github-headings";
import dateFormat from "dateformat";

export default function VicunaEval({ frontmatter, content }) {
  return (
    <div className="w-full flex justify-center py-5 pt-16 md:pt-5">
      <Tags title="Vicuna GPT-4 Evaluation" />
      <div className="container px-5">
        <h1
          lang="en"
          style={{ hyphens: "auto" }}
          className="text-4xl md:text-4xl w-full font-bold break-words"
        >
          {frontmatter.title}
        </h1>
        <hr className="mb-5 mt-2 md:hidden" />
        <p className="text-xl pt-2 pb-2">
          by: {frontmatter.author},{" "}
          {dateFormat(frontmatter.date, "dd mmm, yyyy")}
        </p>
        <hr />
        {/*<div*/}
        {/*  className="article"*/}
        {/*  dangerouslySetInnerHTML={{ __html: md().render(content).replace(/<[^>]+>/g, "") }}*/}
        {/*/>*/}
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

export async function getStaticProps() {
  const fileName = fs.readFileSync(`content/vicuna_eval.md`, "utf-8");
  const { data: frontmatter, content } = matter(fileName);
  return {
    props: {
      frontmatter,
      content,
    },
  };
}
