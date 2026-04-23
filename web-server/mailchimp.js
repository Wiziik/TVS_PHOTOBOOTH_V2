import crypto from "node:crypto";

function dcFromKey(apiKey) {
  const dc = apiKey.split("-")[1];
  if (!dc) throw new Error("MAILCHIMP_API_KEY must end with -<dc> (e.g. -us17)");
  return dc;
}

function authHeader(apiKey) {
  return "Basic " + Buffer.from("any:" + apiKey).toString("base64");
}

function subscriberHash(email) {
  return crypto.createHash("md5").update(email.toLowerCase()).digest("hex");
}

/**
 * Upsert a subscriber on the list with merge fields, then apply a tag.
 * Returns { id, status } on success, throws on failure.
 */
export async function upsertSubscriber({
  apiKey,
  listId,
  email,
  mergeFields = {},
  tag,
  marketingOptIn,
}) {
  if (!apiKey) throw new Error("MAILCHIMP_API_KEY not configured");
  if (!listId) throw new Error("MAILCHIMP_LIST_ID not configured");

  const dc = dcFromKey(apiKey);
  const hash = subscriberHash(email);
  const base = `https://${dc}.api.mailchimp.com/3.0/lists/${listId}`;

  // Not opted in → still store as "transactional" (internal-only, no marketing)
  const status_if_new = marketingOptIn ? "subscribed" : "transactional";

  const putRes = await fetch(`${base}/members/${hash}`, {
    method: "PUT",
    headers: {
      Authorization: authHeader(apiKey),
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email_address: email,
      status_if_new,
      merge_fields: mergeFields,
    }),
  });
  if (!putRes.ok) {
    throw new Error(`Mailchimp PUT member HTTP ${putRes.status}: ${await putRes.text()}`);
  }

  if (tag) {
    const tagRes = await fetch(`${base}/members/${hash}/tags`, {
      method: "POST",
      headers: {
        Authorization: authHeader(apiKey),
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        tags: [{ name: tag, status: "active" }],
      }),
    });
    if (!tagRes.ok) {
      throw new Error(`Mailchimp tag HTTP ${tagRes.status}: ${await tagRes.text()}`);
    }
  }

  return { id: hash, status: status_if_new };
}
