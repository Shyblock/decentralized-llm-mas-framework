const hre = require("hardhat");
const fs = require("fs");
const path = require("path");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);

  const AgentRegistry = await hre.ethers.getContractFactory("AgentRegistry");
  const registry = await AgentRegistry.deploy();
  await registry.waitForDeployment();
  console.log("AgentRegistry deployed to:", registry.target);

  const TaskManager = await hre.ethers.getContractFactory("TaskManager");
  const taskManager = await TaskManager.deploy(registry.target);
  await taskManager.waitForDeployment();
  console.log("TaskManager deployed to:", taskManager.target);

  const IncentiveContract = await hre.ethers.getContractFactory("IncentiveContract");
  const incentive = await IncentiveContract.deploy(registry.target);
  await incentive.waitForDeployment();
  console.log("IncentiveContract deployed to:", incentive.target);

  // Authorize TaskManager (workload writes) and IncentiveContract
  // (reputation/capability writes) as the only callers permitted to
  // mutate agent state on AgentRegistry besides registration itself.
  let tx = await registry.setAuthorizedCaller(taskManager.target, true);
  await tx.wait();
  tx = await registry.setAuthorizedCaller(incentive.target, true);
  await tx.wait();
  console.log("Authorized TaskManager and IncentiveContract on AgentRegistry.");

  const deployedDir = path.join(__dirname, "..", "deployed");
  fs.mkdirSync(deployedDir, { recursive: true });

  const addresses = {
    AgentRegistry: registry.target,
    TaskManager: taskManager.target,
    IncentiveContract: incentive.target,
    deployer: deployer.address,
  };
  fs.writeFileSync(
    path.join(deployedDir, "addresses.json"),
    JSON.stringify(addresses, null, 2)
  );

  // Export ABIs so blockchain_bridge.py doesn't need Hardhat/Node at all.
  const abiOutDir = path.join(deployedDir, "abi");
  fs.mkdirSync(abiOutDir, { recursive: true });

  for (const name of ["AgentRegistry", "TaskManager", "IncentiveContract"]) {
    const artifactPath = path.join(
      __dirname, "..", "artifacts", "contracts", `${name}.sol`, `${name}.json`
    );
    const artifact = JSON.parse(fs.readFileSync(artifactPath));
    fs.writeFileSync(
      path.join(abiOutDir, `${name}.json`),
      JSON.stringify(artifact.abi, null, 2)
    );
  }

  console.log("Saved addresses + ABIs to ./deployed/");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
