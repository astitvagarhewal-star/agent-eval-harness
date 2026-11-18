const { isAdult } = require("../validate");

function testAdult() {
  if (isAdult(18) !== true) throw new Error("18 should be an adult");
  if (isAdult(17) !== false) throw new Error("17 should not be an adult");
}

if (require.main === module) {
  testAdult();
  console.log("ok");
}
