// validate.js: age validation with an off-by-one bug.
function isAdult(age) {
  // BUG: 18-year-olds are incorrectly rejected.
  return age > 18;
}
module.exports = { isAdult };
