import Head from "next/head";
import { useRouter } from "next/router";

export default function Tags(props) {
  const router = useRouter();

  const title = props.title ? props.title + " | LMSYS Org" : "LMSYS Org";
  const desc = props.desc
    ? props.desc
    : "LMSYS Org, Large Model Systems Organization, is an organization missioned to democratize the technologies underlying large models and their system infrastructures.";
  const image = props.image ? props.image : "/social.png";
  const alt = props.alt
    ? props.alt
    : "The text: LLMSYS Org, Large Model Systems Organization.";
  const slug = props.slug ? props.slug : router.route;

  return (
    <Head>
      <title>{title}</title>

      <meta name="title" content={title} />
      <meta property="og:title" content={title} />
      <meta name="twitter:title" content={title} />

      <meta name="description" content={desc} />
      <meta property="og:description" content={desc} />
      <meta name="twitter:description" content={desc} />

      <meta property="og:image" content={"https://lmsys.org" + image} />
      <meta name="twitter:image" content={"https://lmsys.org" + image} />

      <meta name="twitter:image:alt" content={alt} />

      <meta property="og:type" content="website" />
      <meta property="og:url" content={"https://lmsys.org" + slug} />
      <meta name="twitter:url" content={"https://lmsys.org" + slug} />
      <meta name="twitter:card" content="summary_large_image" />

      <meta name="viewport" content="initial-scale=1.0, width=device-width" />
      <meta name="theme-color" content="#1d1d1f" />
      <link rel="icon" type="image/png" sizes="32x32" href="/favicon.jpeg" />
      <link rel="icon" href="/favicon.jpeg" type="image/jpg" />
    </Head>
  );
}
