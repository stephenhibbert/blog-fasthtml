import json
import pathlib

from fasthtml.common import *
from components import *
from contents import *

from datetime import datetime

hdrs = (
    KatexMarkdownJS(),
    HighlightJS(langs=['python', 'javascript', 'html', 'css']),
    Link(rel='stylesheet', href='https://cdn.jsdelivr.net/npm/normalize.css@8.0.1/normalize.min.css', type='text/css'),
    Link(rel='stylesheet', href='https://cdn.jsdelivr.net/npm/sakura.css/css/sakura.css', type='text/css'),    
    Link(rel='stylesheet', href='/public/style.css', type='text/css'),        
)

def not_found(req, exc): return Titled("404: I don't exist!")

exception_handlers = {404: not_found}
app, rt = fast_app(hdrs=hdrs, pico=False, debug=True, exception_handlers=exception_handlers)

@rt("/")
def get():
    posts = [blog_post(title=x["title"],slug=x["slug"],timestamp=x["date"],description=x.get("description", "")) for x in list_posts()]
    popular = [blog_post(title=x["title"],slug=x["slug"],timestamp=x["date"],description=x.get("description", "")) for x in list_posts() if x.get("popular", False)]    
    return Layout(
        Title("Stephen Hibbert"),        
        Socials(site_name="https://stephenhib.com",
                    title="Stephen Hibbert",
                    description="Stephen Hibbert's personal blog",
                    url="https://stephenhib.com",
                    image="https://stephenhib.com/public/images/profile.jpg",
                    ),
        Section(
                H1('Recent Writings'),
                *posts[:3]
            ),
        Hr(),
        Section(
                H1('Popular Writings'),
                *popular
        ),
    )

@rt("/posts")
def get():
    duration = round((datetime.now() - datetime(2024, 6, 1)).days / 365.25, 2)
    description = f'Everything written by Stephen Hibbert in the past {duration} years.'
    posts = [blog_post(title=x["title"],slug=x["slug"],timestamp=x["date"],description=x.get("description", "")) for x in list_posts()]
    return Layout(
        Title("All posts by Stephen Hibbert"),
        Socials(site_name="https://stephenhib.com",
                        title="All posts by Stephen Hibbert",
                        description=description,
                        url="https://stephenhib.com/posts/",
                        image="images/stephenhib.jpeg",
                        ),
        Section(
                H1(f'All Articles ({len(posts)})'),
                P(description),
                *posts,
                A("← Back to home", href="/"),
        ))

@rt("/posts/{slug}")
def get(slug: str):
    if not pathlib.Path(f"posts/{slug}.md").exists():
        raise HTTPException(404)
    # post = [x for x in filter(lambda x: x["slug"] == slug, list_posts())][0]
    content, metadata = get_post(slug)
    # content = pathlib.Path(f"posts/{slug}.md").read_text().split("---")[2]
    # metadata = yaml.safe_load(pathlib.Path(f"posts/{slug}.md").read_text().split("---")[1])    
    tags = [tag(slug=x) for x in metadata.get("tags", [])]
    specials = ()
    return Layout(
        Title(metadata['title']),
        Socials(site_name="https://stephenhib.com",
                        title=metadata["title"],
                        description=metadata.get("description", ""),
                        url=f"https://stephenhib.com/posts/{slug}",
                        image="https://stephenhib.com" + metadata.get("image", default_social_image),
                        ),        
        Section(
            Img(src=metadata.get("image", default_social_image), alt=metadata.get("title", "Stephen Hibbert"), style="width: 100%;"),
            H1(metadata["title"]),
            Div(content,cls="marked"),
            Div(style="width: 200px; margin: auto; display: block;")(*specials),
            P(Span("Tags: "), *tags),
            A("← Back to all articles", href="/"),
        ),
    )

@rt("/tags")
def get():
    tags = [tag_with_count(slug=x[0], count=x[1]) for x in list_tags().items()]
    return Layout(Title("Tags"),
        Socials(site_name="https://stephenhib.com",
                        title="Tags",
                        description="All tags used in the site.",
                        url="https://stephenhib.com/tags/",
                        image="https://stephenhib.com/public/images/profile.jpg",
                        ),               
        Section(
            H1('Tags'),
            P('All tags used in the blog'),
            *tags,
            Br(), Br(),
            A("← Back home", href="/"),
        )
    )

@rt("/tags/{slug}")
def get(slug: str):
    posts = [blog_post(title=x["title"],slug=x["slug"],timestamp=x["date"],description=x.get("description", "")) for x in list_posts() if slug in x.get("tags", [])]
    return Layout(Title(f"Tag: {slug}"),
        Socials(site_name="https://stephenhib.com",
                        title=f"Tag: {slug}",
                        description=f'Posts tagged with "{slug}" ({len(posts)})',
                        url=f"https://stephenhib.com/tags/{slug}",
                        image="https://stephenhib.com/public/images/profile.jpg",
                        ),                       
        Section(
            H1(f'Posts tagged with "{slug}" ({len(posts)})'),
            *posts,
            A("← Back home", href="/"),
        )
    )

@rt("/search")
def get(q: str = ""):
    def _s(obj: dict, name: str, q: str):
        content =  obj.get(name, "")
        if isinstance(content, list):
            content = " ".join(content)
        return q.lower().strip() in content.lower().strip()

    posts = []
    if q:
        posts = [blog_post(title=x["title"],slug=x["slug"],timestamp=x["date"],description=x.get("description", "")) for x in list_posts() if
                    any(_s(x, name, q) for name in ["title", "description", "content", "tags"])]
        
    if posts:
        messages = [H2(f"Search results on '{q}'"), P(f"Found {len(posts)} entries")]
        description = f"Search results on '{q}'. Found {len(posts)} entries"
    elif q:
        messages = [P("No results found for '{q}'")]
        description = f"No results found for '{q}'"
    else:
        messages = []
        description = ""
    return Layout(Title("Search"), 
        Socials(site_name="https://stephenhib.com",
                        title="Search the site",
                        description=description,
                        url="https://stephenhib.com/search",
                        image="https://stephenhib.com/public/images/profile.jpg",
                        ),                    
        Form(Input(name="q", value=q, id="search", type="search", autofocus=True), Button("Search"), style="text-align: center;"),
        Section(
            *messages,
            *posts,
            A("← Back home", href="/"),
        )
    )

reg_re_param("static", "ico|gif|jpg|jpeg|webm|css|js|woff|png|svg|mp4|webp|ttf|otf|eot|woff2|txt")

@rt("/{slug}")
def get(slug: str):
    if pathlib.Path(f"pages/{slug}.md").exists():
        return Layout(*markdown_page(slug))
    raise HTTPException(404)

serve()