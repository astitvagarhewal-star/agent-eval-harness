// coerce.js: equality check with a loose-comparison bug.
function areEqual(a, b) {
  // BUG: loose equality coerces types ("5" == 5).
  return a == b;
}
module.exports = { areEqual };
