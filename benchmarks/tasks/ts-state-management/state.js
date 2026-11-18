// state.js: state management with a merge bug.
function updateState(state, patch) {
  // BUG: wholesale replacement discards existing fields.
  return patch;
}
module.exports = { updateState };
