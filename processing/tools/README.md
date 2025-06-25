# License: http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Discourse Downloader

This Python tool downloads topics and posts from a Discourse server.

## Setup

```bash
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from downloader import DiscourseDownloader

opts = DiscourseDownloader.Options(
    api_key="?api_key=YOUR_API_KEY&api_username=bot",
    verbose=True,
    num_threads=8,
    rate_limit=2.0,          # max 2 requests per second
    save_path="./downloads",  # directory for saving posts
    on_post=lambda t, p, d: print(f"Saved post {p} of topic {t}")
)

dd = DiscourseDownloader("meta.discourse.org", opts)

categories = dd.get_categories()
for cat in categories:
    print(f"{cat['id']}: {cat['name']} ({cat['slug']})")
    for sub in cat.get('subcategories', []):
        print(f"  ↳ {sub['id']}: {sub['name']}")

    topics = dd.get_category_topics(cat['id'])
    if topics:
        print(f"Downloading {len(topics)} topics from {cat['name']}:")
        for topic in topics:
            dd.download_topic(topic)

dd.wait_for_downloads()
```

See requirements.txt for dependencies.
