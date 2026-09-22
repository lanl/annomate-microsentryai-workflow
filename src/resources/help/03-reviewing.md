# Reviewing Images

Each image gets a decision: **Accept** (no defects) or **Reject** (has defects). The Dataset Navigator on the left helps you work through the whole set.

## Make a decision

Use the **Accept** and **Reject** buttons at the top right of the canvas. Click the active button again to clear your decision.

## Review status

The navigator shows how complete each image is:

| Status | Meaning |
|---|---|
| Undecided | No decision yet, and no annotations or tags |
| Reviewed | Accepted with nothing marked, or rejected with supporting work (a polygon in Pixel Level mode, a class tag in Image Level mode) |
| Incomplete | Rejected without any supporting work, or has annotations but no decision yet |
| Conflicting | Accepted, but it also has annotations or tags. Either remove them or change the decision to Reject. |

Hover over an image's status for a plain-language explanation of what is still needed.

## The Dataset Navigator

- Click an image's card to open it. Click **Prev (A)** and **Next (D)** to step through, and check the counter for your position (for example *3 / 120*).
- Each card shows the file name, its decision, and small badges for annotations, inspector and notes. After you run an AI model, it also shows that image's anomaly score.
- Expand a card to see and edit that image's annotations, class tags and notes.

### Filter and sort

Click **Filter** to narrow the list. You can filter by:

- **Decision**: Accept or Reject
- **Status**: Undecided, Reviewed, Incomplete or Conflicting
- **Class**: only images that have annotations in a chosen class

Each option shows how many images match. Under **Sort by**, choose which column orders the list. **Clear filters** resets everything.

## Inspector and notes

Expand an image's card to record who reviewed it and why:

- **Inspector**: the name of the person reviewing. **Set as session inspector** fills the name in automatically on each new image as you go.
- **Image note**: free text about the image.
- To give many images the same inspector, use the bulk option, choose which images it applies to, and check the count shown before confirming.
