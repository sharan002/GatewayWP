// Branch support for the WhatsApp bot: the branch list lives in the shared `configs`
// collection (key: "branches", managed from the CRM); defaults are used until it is saved.
const mongoose = require("mongoose");

const DEFAULT_BRANCHES = ["Coimbatore", "Chennai", "Madurai"];

async function getBranches() {
  try {
    const db = mongoose.connection.db;
    const cfg = db ? await db.collection("configs").findOne({ key: "branches" }) : null;
    if (cfg && Array.isArray(cfg.value) && cfg.value.length) return cfg.value;
  } catch (err) {
    console.error("getBranches error:", err.message);
  }
  return DEFAULT_BRANCHES;
}

module.exports = { DEFAULT_BRANCHES, getBranches };
