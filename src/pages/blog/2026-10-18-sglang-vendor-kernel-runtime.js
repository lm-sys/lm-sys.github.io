import Head from "next/head";
import { useEffect } from "react";

const destination = "/blog/2026-10-18-sglang-low-latency-mode/";

export default function PreviousBlogAddress() {
  useEffect(() => {
    window.location.replace(destination + window.location.search + window.location.hash);
  }, []);

  return (
    <>
      <Head>
        <title>SGLang Low-Latency Mode</title>
        <link rel="canonical" href={`https://www.lmsys.org${destination}`} />
      </Head>
      <p>
        This article has moved to <a href={destination}>SGLang Low-Latency Mode</a>.
      </p>
      <noscript>
        <meta httpEquiv="refresh" content={`0;url=${destination}`} />
      </noscript>
    </>
  );
}
