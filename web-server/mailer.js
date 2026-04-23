/**
 * Placeholder for transactional photo-delivery email.
 *
 * TODO: pick a sender and wire it up. Options:
 *   - Brevo transactional (free 300/day, EU, simple HTTP API)
 *   - Mailchimp Transactional (Mandrill) — requires separate Mailchimp add-on
 *   - SMTP via OVH / Gmail app password / etc (nodemailer)
 *
 * Until then, the app captures emails + subscribes to Mailchimp, logs what
 * *would* be sent, and returns ok. The landing page still works end-to-end.
 */
export async function sendPhotoDeliveryEmail({ to, ticketCode, photoUrls, baseUrl }) {
  console.log(
    `[mailer:stub] deliver ticket=${ticketCode} to=${to} photos=${photoUrls.length} ` +
      `url=${baseUrl}/t/${ticketCode}`,
  );
  return { ok: true, stubbed: true };
}
