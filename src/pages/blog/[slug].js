import fs from "fs";
import matter from "gray-matter";
import md from "markdown-it";
import mdgh from "markdown-it-github-headings";
import mdhl from "markdown-it-highlightjs";
import Tags from "../../components/Tags";
import dateFormat from "dateformat";
import React, { useEffect, useMemo, useRef } from 'react';
import Head from 'next/head';

const COPY_ICON =
  '<svg class="code-copy-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
  'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
  'aria-hidden="true"><rect x="9" y="9" width="13" height="13" rx="2"/>' +
  '<path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>';

const DONE_ICON =
  '<svg class="code-copy-icon-done" width="16" height="16" viewBox="0 0 24 24" fill="none" ' +
  'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" ' +
  'aria-hidden="true"><path d="M20 6 9 17l-5-5"/></svg>';

function renderArticle(content) {
  const renderer = md({ html: true })
    .use(mdgh, { prefixHeadingIds: false })
    .use(mdhl);

  // Wrap every fenced block so the copy button has a positioning context.
  // `pre` itself scrolls horizontally, so a button inside it would drift
  // out of view as soon as the reader scrolls a long line.
  const renderFence = renderer.renderer.rules.fence;
  renderer.renderer.rules.fence = (tokens, idx, options, env, self) => {
    const code = renderFence(tokens, idx, options, env, self);
    return (
      '<div class="code-block">' +
      '<button class="code-copy" type="button" aria-label="Copy code">' +
      COPY_ICON +
      DONE_ICON +
      "</button>" +
      code +
      "</div>"
    );
  };

  return renderer.render(content);
}

async function copyText(text) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (err) {
    // Permission denied or no clipboard access: fall through to the
    // selection-based path below.
  }

  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  area.style.position = "fixed";
  area.style.top = "0";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();

  let copied = false;
  try {
    copied = document.execCommand("copy");
  } catch (err) {
    copied = false;
  }
  document.body.removeChild(area);
  return copied;
}

export default function Post({ frontmatter, content, slug }) {
  const articleRef = useRef(null);
  const html = useMemo(() => renderArticle(content), [content]);

  useEffect(() => {
    // Check if MathJax object is available
    if (window.MathJax) {
      // Typeset / reprocess the math content
      window.MathJax.typesetPromise();
    }
  }, [content]);

  useEffect(() => {
    const article = articleRef.current;
    if (!article) return;

    // The article is injected as raw HTML, so the buttons are not React
    // nodes. One delegated listener covers every code block on the page.
    const timers = new Map();

    const onClick = async (event) => {
      const button = event.target.closest(".code-copy");
      if (!button || !article.contains(button)) return;

      const code = button.parentElement.querySelector("pre code");
      if (!code) return;

      if (!(await copyText(code.textContent))) return;

      button.dataset.copied = "true";
      clearTimeout(timers.get(button));
      timers.set(
        button,
        setTimeout(() => {
          delete button.dataset.copied;
          timers.delete(button);
        }, 2000)
      );
    };

    article.addEventListener("click", onClick);
    return () => {
      article.removeEventListener("click", onClick);
      timers.forEach((id) => clearTimeout(id));
    };
  }, [html]);

  return (
    <div className="w-full flex justify-center py-5 pt-16 md:pt-5">
      <Head>
        {/* Add highlight.js CSS for styling code blocks */}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.5.0/styles/atom-one-light.min.css" />
      </Head>
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
          ref={articleRef}
          className="pt-2 article"
          dangerouslySetInnerHTML={{ __html: html }}
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
