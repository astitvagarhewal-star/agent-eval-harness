const { areEqual } = require("../coerce");

function testStrict() {
  if (areEqual("5", 5) !== false) throw new Error("'5' should not equal 5");
  if (areEqual(5, 5) !== true) throw new Error("5 should equal 5");
}

if (require.main === module) {
  testStrict();
  console.log("ok");
}
