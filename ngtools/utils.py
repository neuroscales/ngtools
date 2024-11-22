"""Various utilities."""
# stdlib
import json
from urllib.parse import quote

# externals
import neuroglancer as ng

DEFAULT_URL = (
    "https://neuroglancer.lincbrain.org/cloudfront/frontend/index.html"
)


def neuroglancer_state_to_neuroglancer_url(
    state: ng.ViewerState,
    base_url: str = DEFAULT_URL,
) -> str:
    """
    Take the current state of the Neuroglancer viewer and convert it
    into a url-encoded value for the browser.
    """
    ordered_dict = state.to_json()

    # Convert the dictionary to a JSON string
    json_str = json.dumps(ordered_dict)

    # URL-encode the JSON string
    encoded_json = quote(json_str)

    # Construct and return the full URL
    return f"{base_url}#!{encoded_json}"
