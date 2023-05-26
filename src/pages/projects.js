import Tags from "../components/Tags";
import Link from "next/link";
import projects from "../../content/projects.json";

export default function Home() {
  return (
    <div className="w-full flex justify-center py-5 pt-16 md:pt-5">
      <Tags
        title="Projects"
        desc="LMSYS Org develops open datasets, models, systems, and evaluation tools for large models."
      />
      <div className="container px-5">
        <h1 className="text-7xl md:text-8xl font-bold">PROJECTS</h1>
        <div className="text-2xl pb-4">
            LMSYS Org develops open datasets, models, systems, and evaluation tools for large models.
        </div>
        <hr className="mb-5 mt-2 md:hidden" />
        {projects.map((item, i) => {
          return (
            <div key={i}>
              <h3 className={"pb-4 " + (i === 0 ? "pt-0" : "pt-5")}>
                {item.name.toUpperCase()}
              </h3>
              <div className="grid gap-5 grid-cols-2">
                {item.entries.map((item, i) => {
                  if (item.link.charAt(0) === `/`) {
                    return (
                      <a
                        key={item.link}
                        href={item.link}
                        className={
                          "no-underline " +
                          (item.desc === undefined
                            ? "col-span-1"
                            : "col-span-2")
                        }
                      >
                        <Link key={i} href={item.link}>
                          <ProjectItem item={item} />
                        </Link>
                      </a>
                    );
                  } else {
                    return (
                      <a
                        key={i}
                        href={item.link}
                        rel="noopener noreferrer"
                        target="_blank"
                        className={
                          "no-underline " +
                          (item.desc === undefined
                            ? "col-span-1"
                            : "col-span-2")
                        }
                      >
                        <ProjectItem item={item} />
                      </a>
                    );
                  }
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ProjectItem({ item }) {
  return (
    <div
      className={
        " bg-sky text-paper border border-paper hover:bg-paper hover:text-sky cursor-pointer transition-colors p-5 shadow-lg shadow-neutral-800/20 flex flex-col sm:flex-row "
      }
    >
      <div className={item.desc !== undefined && "basis-1/4"}>
        <p className="text-2xl">{item.name}</p>
        <p className="text-sm">{item.architecture}</p>
        <p className="text-sm">{item.size}</p>
      </div>
      <hr
        className={
          "mt-4 mb-4 sm:hidden " + (item.desc === undefined && "hidden")
        }
      />
      <div className={"text-lg basis-3/4 " + (item.desc === undefined && "hidden")}>
        {item.desc}
      </div>
    </div>
  );
}
