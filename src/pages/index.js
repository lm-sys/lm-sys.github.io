import Tags from "../components/Tags";
import Link from "next/link";
import { Fade } from "react-slideshow-image";
import { useEffect } from "react";
import { useRouter } from "next/router";
import "react-slideshow-image/dist/styles.css";

export default function Home() {
  const router = useRouter();
  useEffect(() => {
    if (location.hash === "#contact") {
      router.push("/gdpr");
    }
  }, []);

  return (
    <>
      <Slider className="md:hidden" />
      <div className="pt-20 md:pt-0 full-container md:h-screen w-full md:flex flex-col items-center justify-center text-center relative child:absolute">
        <Slider className="hidden md:block" />
        <Tags />
        <div className="flex items-center justify-center w-full flex-col">
          <div className="w-auto md:pb-0 flex flex-col md:flex-row items-center gap-4 px-4 overflow-x-hidden">
            <div className="bg-sky text-paper border-white border p-3">
              <div className="flex justify-center">
                <h1 className="text-8xl md:text-8xl font-bold">LMSYS Org</h1>
                {/*<div className="flex items-center">*/}
                {/*  <img src={"/images/logo.svg"} className="pl-4 pb-1 h-16" />*/}
                {/*</div>*/}
              </div>
              <p className="italic pb-3">
              </p>
              <p className="pt-2 py-4 max-w-lg flex m-auto">
                  The Large Model Systems Organization develops large models and systems that are open, accessible, and scalable.
              </p>
            </div>

            <div
              className="w-fit h-fit items-center grid sm:grid-cols-2 gap-4
            child:h-full child:flex child:flex-col child:justify-center
                      child:border child:border-paper child:bg-sky child:text-paper child:p-3 child:max-w-[16rem] hover:child:bg-paper hover:child:text-sky child:transition-colors child:cursor-pointer child:mx-auto md:child:mx-0"
            >
              <Link href="/blog/2023-03-30-vicuna">
                <div className="">
                  <p className="text-2xl pb-1">Vicuna</p>
                  <hr />
                  <p className="pt-2">
                    A chatbot impressing GPT-4 with 90%* ChatGPT quality, available in 7B/13B/33B sizes.
                  </p>
                </div>
              </Link>

              <Link href="https://chat.lmsys.org/">
                <div className="">
                  <p className="text-2xl pb-1">Chatbot Arena</p>
                  <hr />
                  <p className="pt-2">
                    Scalable and gamified evaluation of LLMs via crowdsourcing and Elo rating systems.
                  </p>
                </div>
              </Link>

              <Link href="https://github.com/lm-sys/FastChat">
                <div className="">
                  <p className="text-2xl pb-1">FastChat</p>
                  <hr />
                  <p className="pt-2">
                    An open platform for training, serving, and evaluating LLM-based chatbots.
                  </p>
                </div>
              </Link>

              <Link href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">
                <div className="">
                  <p className="text-2xl pb-1">LMSYS-Chat-1M</p>
                  <hr />
                  <p className="pt-2">
                    A large-scale real-world LLM conversation dataset.
                  </p>
                </div>
              </Link>

              <Link href="https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge">
                <div className="">
                  <p className="text-2xl pb-1">MT-Bench</p>
                  <hr />
                  <p className="pt-2">
                    A set of challenging, multi-turn, and open-ended questions for evaluating chatbots.
                  </p>
                </div>
              </Link>

              <Link href="https://github.com/DachengLi1/LongChat">
                <div className="">
                  <p className="text-2xl pb-1">LongChat</p>
                  <hr />
                  <p className="pt-2">
                    A chatbot supporting 16K context length, available in 7B and 13B sizes.
                  </p>
                </div>
              </Link>

            </div>
          </div>
        </div>
      </div>
    </>
  );
}

function Slider({ className }) {
  // const slideImages = ["dragon", "painting", "fox", "river"];
  const slideImages = ["universe", "matrix", "llama", "liandan"];

  return (
    <div className={"z-0 h-screen w-full fixed md:absolute " + className}>
      <div className="slide-container">
        <Fade duration={3500} arrows={false} pauseOnHover={false}>
          {slideImages.map((slideImage, index) => {
            return (
              <picture key={index}>
                {/*<source*/}
                {/*  srcSet={"/images/gallery/" + slideImage + ".avif"}*/}
                {/*  type="image/avif"*/}
                {/*/>*/}
                {/*<source*/}
                {/*  srcSet={"/images/gallery/" + slideImage + ".webp"}*/}
                {/*  type="image/webp"*/}
                {/*/>*/}
                {/*<source*/}
                {/*  srcSet={"/images/gallery/" + slideImage + ".png"}*/}
                {/*  type="image/png"*/}
                {/*/>*/}
                <img
                  src={"/images/gallery/" + slideImage + ".png"}
                  className="object-cover  h-full w-full opacity-20 md:opacity-25"
                />
              </picture>
            );
          })}
        </Fade>
      </div>
    </div>
  );
}
