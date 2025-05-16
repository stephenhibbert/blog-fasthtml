from fasthtml.common import *
from datetime import datetime
from dateutil import parser
import functools
import pathlib
import pytz
import yaml
import collections
from nb2fasthtml.core import (
    render_nb, read_nb, get_frontmatter_raw,render_md,
    strip_list
)

profile_pic = "/public/images/profile.jpg"


class ContentNotFound(Exception):
    pass

search_modal_css = Style(
    """
.modal {
  position: fixed;
  z-index: 1;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background-color: rgba(0,0,0,0.4);
}
.modal-content {
  background-color: #f9f9f9;
  margin: 15% auto;
  padding: 20px;
  border: 1px solid #888;
  width: 80%;
  max-width: 600px;
}
#search-input {
  width: 100%;
  padding: 10px;
  margin-bottom: 10px;
}
a {color: #059669 !important;}
h1 {
    font-size: 2.5rem;
    line-height: 1.2;
    font-weight: 800;
    letter-spacing: -0.05rem;
    margin: 1rem 0;
}
h2 {
    font-size: 2rem;
    line-height: 1.3;
    font-weight: 800;
    letter-spacing: -0.05rem;
    margin: 1rem 0;
}
h3 {
    font-size: 1.5rem;
    line-height: 1.4;
    margin: 1rem 0;
}
h4 {
    font-size: 1.2rem;
    line-height: 1.5;
}
.borderCircle {
    border-radius: 9999px;
    margin-bottom: 0rem;
    text-decoration: none;
}
.list {
    list-style: none;
    padding: 0;
    margin: 0;
}
.listItem {
    margin: 0 0 1.25rem;
}
.lightText {
    color: #666;
}
.center {
    display: flex;
    justify-content: center;
}
/* Add these new styles for a moderately wider content area */
body {
    max-width: 80% !important;
    margin: 0 auto !important;
}
main {
    width: 100% !important;
    max-width: 900px !important;  /* More moderate width */
    margin: 0 auto !important;
}
"""
)

hdrs = (
    KatexMarkdownJS(),
    HighlightJS(langs=["python", "javascript", "html", "css"]),
    Link(
        rel="icon",
        href="/public/favicon.ico",
        type="image/x-icon"
    ),
    Link(
        rel="shortcut icon",
        href="/public/favicon.ico",
        type="image/x-icon"
    ),
    Link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/normalize.css@8.0.1/normalize.min.css",
        type="text/css",
    ),
    Link(
        rel="stylesheet",
        href="https://cdn.jsdelivr.net/npm/sakura.css/css/sakura.css",
        type="text/css",
    ),
    search_modal_css
)


def convert_dtstr_to_dt(date_str: str) -> datetime:
    """Convert a naive or non-naive date/datetime string to a datetime object."""
    if not date_str:
        return None

    try:
        dt = parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=pytz.UTC)
        return dt
    except (ValueError, TypeError) as e:
        print(f"Warning: Could not parse date string: {date_str}, error: {e}")
        # Return a default date instead of None
        return datetime(1970, 1, 1, tzinfo=pytz.UTC)  # Unix epoch as fallback

def format_datetime(dt: datetime) -> str:
    """Format the datetime object in a consistent way."""
    formatted_date = dt.strftime("%B %d, %Y")
    formatted_time = dt.strftime("%I:%M%p").lstrip("0").lower()
    return f"{formatted_date} at {formatted_time}"


def render_code_output(cell, lang='python', render_md=render_md):
    import re

    def escape_backticks(text):
        # Replace backticks with escaped version
        return str(text).replace('`', '\\`')

    def strip_ansi_codes(text):
        # Regular expression to remove ANSI escape sequences
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        return ansi_escape.sub('', str(text))

    def handle_image_data(data):
        # Handle different image formats - existing code...
        if 'image/png' in data:
            img_data = data['image/png']
            return f'![png](data:image/png;base64,{img_data})'
        elif 'image/jpeg' in data:
            img_data = data['image/jpeg']
            return f'![jpeg](data:image/jpeg;base64,{img_data})'
        elif 'image/svg+xml' in data:
            img_data = data['image/svg+xml']
            return f'<img src="data:image/svg+xml;base64,{base64.b64encode(img_data.encode()).decode()}">'
        return None

    res = []
    if len(cell['outputs']) == 0:
        return ''

    for output in cell['outputs']:
        if output['output_type'] == 'execute_result':
            data = output['data']
            if 'text/markdown' in data:
                res.append(NotStr(''.join(strip_list(data['text/markdown'][1:-1]))))
            elif 'text/plain' in data:
                plain_text = ''.join(strip_list(data['text/plain']))
                res.append(strip_ansi_codes(plain_text))  # Strip ANSI codes
            # Handle potential image output in execute_result
            img_output = handle_image_data(data)
            if img_output:
                res.append(img_output)

        elif output['output_type'] == 'stream':
            stream_text = ''.join(strip_list(output['text']))
            res.append(strip_ansi_codes(stream_text))  # Strip ANSI codes

        elif output['output_type'] == 'display_data':
            # Handle display_data type which is commonly used for images
            data = output['data']
            img_output = handle_image_data(data)
            if img_output:
                res.append(img_output)
            elif 'text/plain' in data:
                plain_text = ''.join(strip_list(data['text/plain']))
                res.append(strip_ansi_codes(plain_text))  # Strip ANSI codes

        elif output['output_type'] == 'error':
            # Handle error outputs
            error_msg = '\n'.join([strip_ansi_codes(line) for line in output['traceback']])  # Strip ANSI codes
            res.append(f"Error:\n{error_msg}")

    # Combine all results into a single string with newlines between them
    if res:
        combined_output = '\n'.join(escape_backticks(r) for r in res)
        return render_md(combined_output, container=Pre)
    return ''


def render_mermaid(graph):
    """
    Renders a Mermaid diagram by converting it to a base64-encoded image URL
    """
    import base64

    # Encode the graph definition
    graph_bytes = graph.encode("utf8")
    base64_bytes = base64.urlsafe_b64encode(graph_bytes)
    base64_string = base64_bytes.decode("ascii")

    # Return markdown image syntax
    return f'![mermaid](https://mermaid.ink/img/{base64_string})'


@functools.cache
def list_posts(published: bool = True, posts_dirname="posts", content=False) -> list[dict]:
    """
    Loads all the posts and their frontmatter.
    Note: Could use pathlib better
    """
    posts: list[dict] = []
    # Fetch notebooks
    for post in pathlib.Path('.').glob(f"{posts_dirname}/**/*.ipynb"):
        if '.ipynb_checkpoints' in str(post): continue
        nb = read_nb(post)
        data: dict = get_frontmatter_raw(nb.cells[0])
        data["slug"] = post.stem
        data['cls'] = 'notebook'
        if content:
            data["content"] = render_nb(post,
                                        cls='',
                                        fm_fn=lambda x: '',
                                        out_fn=render_code_output
                                        )
        posts.append(data)
    # Fetch markdown
    for post in pathlib.Path('.').glob(f"{posts_dirname}/**/*.md"):
        raw: str = post.read_text().split("---")[1]
        data: dict = yaml.safe_load(raw)
        data["slug"] = post.stem
        data['cls'] = 'marked'
        if content:
            data["content"] = '\n'.join(post.read_text().split("---")[2:])
        posts.append(data)

    # Create a copy of the posts for sorting
    sorted_posts = []
    for post in posts:
        post_copy = post.copy()  # Create a copy to avoid modifying the original
        # Convert the date string to a datetime object for sorting
        if isinstance(post_copy["date"], str):
            dt_obj = convert_dtstr_to_dt(post_copy["date"])
            if dt_obj:  # Only add if conversion was successful
                post_copy["_sort_date"] = dt_obj  # Add a separate field for sorting
                sorted_posts.append(post_copy)
            else:
                print(
                    f"Warning: Could not parse date for post {post_copy.get('title', 'Unknown')}: {post_copy['date']}")
        else:
            # If it's already a datetime object
            post_copy["_sort_date"] = post_copy["date"]
            sorted_posts.append(post_copy)

    # Sort based on the datetime objects
    sorted_posts.sort(key=lambda x: x["_sort_date"], reverse=True)

    # Filter for published posts
    return [x for x in sorted_posts if x["published"] is published]


@functools.lru_cache
def get_post(slug: str) -> tuple:
    """Get a specific post with caching."""
    posts = list_posts(content=True)
    post = next((x for x in posts if x["slug"] == slug), None)
    if post is None:
        raise ContentNotFound
    return (post["content"], post)


@functools.cache
def list_tags() -> dict[str, int]:
    """List all tags with their counts, cached."""
    unsorted_tags = {}
    for post in list_posts():
        for tag in post.get("tags", []):
            unsorted_tags[tag] = unsorted_tags.get(tag, 0) + 1
    return collections.OrderedDict(
        sorted(unsorted_tags.items(), key=lambda x: x[1], reverse=True)
    )


# Components
def Layout(title, socials, *tags):
    """Enhanced layout with improved navigation and search."""
    return (
        title,
        socials,
        (
            Header(
                A(
                    Img(
                        cls="borderCircle",
                        alt="Stephen Hibbert",
                        src="/public/images/profile.jpg",
                        width="108",
                        height="108",
                    ),
                    href="/",
                ),
                A(H2("Stephen Hibbert"), href="/"),
                P(
                    A("About", href="/about"),
                    " | ",
                    A("Articles", href="/posts"),
                    " | ",
                    A("Tags", href="/tags"),
                    " | ",
                    A("Search", href="/search"),
                ),
                style="text-align: center;",
            ),
            Main(*tags),
            Footer(
                Hr(),
                P(
                    A(
                        "LinkedIn",
                        href="https://www.linkedin.com/in/stephen-hibbert-2b7a045b/",
                    ),
                    " | ",
                    A("Twitter", href="https://twitter.com/stephenhib"),
                ),
                P(f"All rights reserved {datetime.now().year}, Stephen Hibbert"),
            ),
            Div(
                Div(
                    H2("Search"),
                    Input(
                        name="q",
                        type="text",
                        id="search-input",
                        hx_trigger="keyup",
                        placeholder="Enter your search query...",
                        hx_get="/search-results",
                        hx_target="#search-results",
                    ),
                    Div(id="search-results"),
                    cls="modal-content",
                ),
                id="search-modal",
                style="display:none;",
                cls="modal",
            ),
            Script(
                """
        document.body.addEventListener('keydown', e => {
            if (e.key === '/' && e.target.tagName !== 'INPUT') {
                e.preventDefault();
                document.getElementById('search-modal').style.display = 'block';
                document.getElementById('search-input').focus();
            }
            if (e.key === 'Escape') {
                document.getElementById('search-modal').style.display = 'none';
            }
        });
        """
            ),
        ),
    )

def BlogPostPreview(title: str, slug: str, timestamp: str, description: str, image: str):
    """
    Enhanced blog post preview with image thumbnail to the left of post info.
    
    Parameters:
    - title: The blog post title
    - slug: URL slug for the post link
    - timestamp: ISO format date string
    - description: Short preview text
    - image: Path to the thumbnail image
    """
    # Format timestamp (assuming you have these functions defined elsewhere)
    # If you don't have these functions, replace this with your preferred date formatting
    try:
        formatted_date = timestamp  # Use your formatting functions here if available
    except:
        formatted_date = timestamp
    
    # Create the article container with flex layout
    return Article(
        # Container div with flexbox to place image next to content
        Div(
            # Left side - Image container with fixed dimensions
            Div(
                Img(src=image, alt=f"Thumbnail for {title}"),
                style="width: 120px; height: 120px; overflow: hidden; margin-right: 15px;"
            ),
            
            # Right side - Content container
            Div(
                H2(A(title, href=f"/posts/{slug}"), style="margin-top: 0; margin-bottom: 8px;"),
                P(description, style="margin-bottom: 8px;"),
                P(Time(formatted_date), style="font-size: 0.8em; color: #666;", cls="timestamp")
            ),
            
            # Styles for the flex container
            style="display: flex; margin-bottom: 20px; align-items: flex-start;"
        )
    )


def TagLink(slug: str):
    return Span(A(slug, href=f"/tags/{slug}"), " ")


def TagLinkWithCount(slug: str, count: int):
    """Render a tag link with post count."""
    return A(Span(f"{slug}"), Small(f" ({count}) "), href=f"/tags/{slug}")


def MarkdownPage(slug: str):
    """Render a markdown page."""
    try:
        text = pathlib.Path(f"pages/{slug}.md").read_text()
    except FileNotFoundError:
        raise ContentNotFound
    content = "".join(text.split("---")[2:])
    metadata = yaml.safe_load(text.split("---")[1])
    return (
        Title(metadata.get("title", slug)),
        Socials(
            site_name="https://stephenhib.com",
            title=metadata.get("title", slug),
            description=metadata.get("description", "slug"),
            url=f"https://stephenhib.com/{slug}",
            image=metadata.get("image", profile_pic),
        ),
        A("← Back to home", href="/"),
        Section(Div(content, cls="marked")),
    )


# Search functionality
def _search(q: str = ""):
    """Internal search function."""

    def _s(obj: dict, name: str, q: str):
        content = obj.get(name, "")
        if isinstance(content, list):
            content = " ".join(content)
        return q.lower().strip() in content.lower().strip()

    messages = []
    posts = []
    description = f"No results found for '{q}'"

    if q.strip():
        posts = [
            BlogPostPreview(
                title=x["title"],
                slug=x["slug"],
                timestamp=x["date"],
                description=x.get("description", ""),
            )
            for x in list_posts()
            if any(
                _s(x, name, q) for name in ["title", "description", "content", "tags"]
            )
        ]

    if posts:
        messages = [H2(f"Search results for '{q}'"), P(f"Found {len(posts)} entries")]
        description = f"Search results for '{q}'"
    elif q.strip():
        messages = [P(f"No results found for '{q}'")]

    return Div(
        Meta(property="description", content=description),
        Meta(property="og:description", content=description),
        Meta(name="twitter:description", content=description),
        *messages,
        *posts,
    )


# Routes
def not_found(request=None, exc=None):
    """404 error handler."""
    return Layout(
        Title("404: Page Not Found"),
        Socials(
            site_name="https://stephenhib.com",
            title="Page Not Found",
            description="The page you're looking for doesn't exist.",
            url="https://stephenhib.com",
            image=profile_pic,
        ),
        H1("404: Page Not Found"),
        P("The page you're looking for doesn't exist."),
        A("← Back to home", href="/"),
    )


exception_handlers = {404: not_found}
app, rt = fast_app(hdrs=hdrs, pico=False, exception_handlers=exception_handlers, live=True)


@rt("/")
def index():
    """Home page route."""
    posts = [
        BlogPostPreview(
            title=x["title"],
            slug=x["slug"],
            timestamp=x["date"],
            description=x.get("description", ""),
        )
        for x in list_posts()
    ]
    # popular = [
    #     BlogPostPreview(
    #         title=x["title"],
    #         slug=x["slug"],
    #         timestamp=x["date"],
    #         description=x.get("description", ""),
    #     )
    #     for x in list_posts()
    #     if x.get("popular", False)
    # ]

    return Layout(
        Title("Stephen Hibbert"),
        Socials(
            site_name="https://stephenhib.com",
            title="Stephen Hibbert",
            description="Stephen Hibbert's personal blog",
            url="https://stephenhib.com",
            image="https://stephenhib.com/public/images/profile.jpg",
        ),
        Section(H1("Recent Writings"), *posts[:3]),
        # Hr(),
        # Section(H1("Popular Writings"), *popular),
    )


@rt("/posts")
def posts():
    """All posts page route."""
    duration = round((datetime.now() - datetime(2024, 6, 1)).days / 365.25, 2)
    description = f"Everything written by Stephen Hibbert in the past {duration} years."
    posts = [
        BlogPostPreview(
            title=x["title"],
            slug=x["slug"],
            timestamp=x["date"],
            description=x.get("description", ""),
            image=x.get("image")
        )
        for x in list_posts()
    ]

    return Layout(
        Title("All posts by Stephen Hibbert"),
        Socials(
            site_name="https://stephenhib.com",
            title="All posts by Stephen Hibbert",
            description=description,
            url="https://stephenhib.com/posts/",
            image="https://stephenhib.com/public/images/profile.jpg",
        ),
        Section(
            H1(f"All Articles ({len(posts)})"),
            P(description),
            *posts,
            A("← Back to home", href="/"),
        ),
    )


@rt("/posts/{slug}")
def get(slug: str):
    """Individual post page route."""
    try:
        content, metadata = get_post(slug)
    except ContentNotFound:
        raise HTTPException(404)

    tags = [TagLink(slug=x) for x in metadata.get("tags", [])]

    return Layout(
        Title(metadata["title"]),
        Socials(
            site_name="https://stephenhib.com",
            title=metadata["title"],
            description=metadata.get("description", ""),
            url=f"https://stephenhib.com/posts/{slug}",
            image="https://stephenhib.com"
            + metadata.get("image", profile_pic),
        ),
        Section(
            H1(metadata["title"]),
            Div(content, cls=metadata["cls"]),
            P(Span("Tags: "), *tags),
            A("← Back to all articles", href="/"),
        ),
    )


@rt("/tags")
def tags():
    """Tags page route."""
    tags = [TagLinkWithCount(slug=x[0], count=x[1]) for x in list_tags().items()]
    return Layout(
        Title("Tags"),
        Socials(
            site_name="https://stephenhib.com",
            title="Tags",
            description="All tags used in the site.",
            url="https://stephenhib.com/tags/",
            image="https://stephenhib.com/public/images/profile.jpg",
        ),
        Section(
            H1("Tags"),
            P("All tags used in the blog"),
            *tags,
            Br(),
            Br(),
            A("← Back home", href="/"),
        ),
    )


@rt("/tags/{slug}")
def tag(slug: str):
    """Individual tag page route."""
    posts = [
        BlogPostPreview(
            title=x["title"],
            slug=x["slug"],
            timestamp=x["date"],
            description=x.get("description", ""),
        )
        for x in list_posts()
        if slug in x.get("tags", [])
    ]

    return Layout(
        Title(f"Tag: {slug}"),
        Socials(
            site_name="https://stephenhib.com",
            title=f"Tag: {slug}",
            description=f'Posts tagged with "{slug}" ({len(posts)})',
            url=f"https://stephenhib.com/tags/{slug}",
            image="https://stephenhib.com/public/images/profile.jpg",
        ),
        Section(
            H1(f'Posts tagged with "{slug}" ({len(posts)})'),
            *posts,
            A("← Back home", href="/"),
        ),
    )


@rt("/search")
def search(q: str = ""):
    """Search page route."""
    return Layout(
        Title("Search"),
        Socials(
            site_name="https://stephenhib.com",
            title="Search the site",
            description="Search through all articles",
            url="https://stephenhib.com/search",
            image="https://stephenhib.com/public/images/profile.jpg",
        ),
        Form(style="text-align: center;")(
            Input(name="q", id="q", value=q, type="search", autofocus=True),
            Button(
                "Search",
                hx_get="/search-results",
                hx_target="#search-results",
                hx_include="#q",
                onclick="updateQinURL()",
            ),
        ),
        Section(
            Div(id="search-results")(_search(q) if q else ""),
            A("← Back home", href="/"),
        ),
        Script(
            """
            function updateQinURL() {
                let url = new URL(window.location);
                const value = document.getElementById('q').value
                url.searchParams.set('q', value);
                window.history.pushState({}, '', url);            
            }
        """
        ),
    )


@rt("/search-results")
def search_results(q: str):
    """HTMX-powered search results route."""
    return _search(q)


@rt("/{slug}")
def page(slug: str):
    """Generic page route for markdown pages."""
    try:
        return Layout(*MarkdownPage(slug))
    except ContentNotFound:
        raise HTTPException(404)


# Static files route
reg_re_param(
    "static", "ico|gif|jpg|jpeg|webm|css|js|woff|png|svg|mp4|webp|ttf|otf|eot|woff2|txt"
)

if __name__ == "__main__":
    serve(reload_includes="*.md,*.ipynb")
