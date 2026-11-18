// parseints.js: comma-separated int parser with an empty-field bug.
function parseInts(input) {
  // BUG: empty fields become 0 instead of being skipped.
  return input.split(",").map(Number);
}
module.exports = { parseInts };
