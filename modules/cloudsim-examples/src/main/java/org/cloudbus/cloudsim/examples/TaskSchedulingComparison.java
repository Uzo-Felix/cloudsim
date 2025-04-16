package org.cloudbus.cloudsim.examples;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.provisioners.BwProvisionerSimple;
import org.cloudbus.cloudsim.provisioners.PeProvisionerSimple;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.provisioners.RamProvisionerSimple;

import java.text.DecimalFormat;
import java.util.*;

public class TaskSchedulingComparison {
    private static List<Cloudlet> cloudletList;
    private static List<Vm> vmList;
    private static DatacenterBroker broker;
    
    // ACO parameters
    private static final double ALPHA = 1.0; // Pheromone importance
    private static final double BETA = 2.0;  // Heuristic importance
    private static final double RHO = 0.1;   // Evaporation rate
    private static final double Q = 100.0;   // Pheromone deposit factor
    private static Map<Integer, Map<Integer, Double>> pheromoneMatrix; // Cloudlet-VM pheromone trails

    public static void main(String[] args) {
        Log.printLine("Starting Task Scheduling Comparison...");
    
        try {
            // Initialize CloudSim only once
            int numUsers = 1;
            Calendar calendar = Calendar.getInstance();
            boolean traceFlag = false;
            CloudSim.init(numUsers, calendar, traceFlag);
    
            // Create Datacenter (only once)
            Datacenter datacenter = createDatacenter("Datacenter_0");
    
            // Create initial broker with proper exception handling
            try {
                broker = new DatacenterBroker("Broker");
                int brokerId = broker.getId();
    
                // Create VMs and Cloudlets (only once)
                vmList = createVms(brokerId, 4); // Reduced to 4 VMs to fit host capacity
                cloudletList = createCloudlets(brokerId, 10);
                initializePheromoneMatrix();
    
                // Run Round Robin Scheduling
                runRoundRobinScheduling();
                
                // Reinitialize CloudSim for next algorithm
                CloudSim.init(numUsers, calendar, traceFlag);
                datacenter = createDatacenter("Datacenter_0");
                broker = new DatacenterBroker("Broker");
                brokerId = broker.getId();
                
                // Run Genetic Algorithm Scheduling
                runGeneticAlgorithmScheduling();
                
                // Reinitialize CloudSim for next algorithm
                CloudSim.init(numUsers, calendar, traceFlag);
                datacenter = createDatacenter("Datacenter_0");
                broker = new DatacenterBroker("Broker");
                brokerId = broker.getId();
                
                // Run ACO Scheduling
                runACOScheduling();
    
                Log.printLine("Task Scheduling Comparison finished!");
            } catch (Exception e) {
                Log.printLine("Failed to create broker: " + e.getMessage());
                e.printStackTrace();
            }
        } catch (Exception e) {
            e.printStackTrace();
            Log.printLine("Error during simulation initialization");
        }
    }
    

    private static void initializePheromoneMatrix() {
        pheromoneMatrix = new HashMap<>();
        double initialPheromone = 1.0;
        
        for (Cloudlet cloudlet : cloudletList) {
            Map<Integer, Double> vmPheromones = new HashMap<>();
            for (Vm vm : vmList) {
                vmPheromones.put(vm.getId(), initialPheromone);
            }
            pheromoneMatrix.put(cloudlet.getCloudletId(), vmPheromones);
        }
    }

    private static Datacenter createDatacenter(String name) {
        List<Host> hostList = new ArrayList<>();
        int mips = 1000;
        int ram = 2048; // MB
        long storage = 1000000; // MB
        int bw = 10000; // Mbps
        int pesNumber = 4; // Number of CPUs

        List<Pe> peList = new ArrayList<>();
        for (int i = 0; i < pesNumber; i++) {
            peList.add(new Pe(i, new PeProvisionerSimple(mips)));
        }

        hostList.add(new Host(
                0, // Host ID
                new RamProvisionerSimple(ram),
                new BwProvisionerSimple(bw),
                storage,
                peList,
                new VmSchedulerTimeShared(peList)
        ));

        String arch = "x86";
        String os = "Linux";
        String vmm = "Xen";
        double timeZone = 10.0;
        double costPerSec = 3.0;
        double costPerMem = 0.05;
        double costPerStorage = 0.001;
        double costPerBw = 0.0;

        DatacenterCharacteristics characteristics = new DatacenterCharacteristics(
                arch, os, vmm, hostList, timeZone, costPerSec, costPerMem, costPerStorage, costPerBw);

        try {
            return new Datacenter(name, characteristics, new VmAllocationPolicySimple(hostList), new LinkedList<>(), 0);
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
    private static List<Vm> createVms(int brokerId, int count) {
        List<Vm> vms = new ArrayList<>();
        int mips = 1000;
        int ram = 512; // MB
        long bw = 1000; // Mbps
        long size = 10000; // MB
        String vmm = "Xen";

        for (int i = 0; i < count; i++) {
            vms.add(new Vm(i, brokerId, mips, 1, ram, bw, size, vmm, new CloudletSchedulerTimeShared()));
        }
        return vms;
    }

    private static List<Cloudlet> createCloudlets(int brokerId, int count) {
        List<Cloudlet> cloudlets = new ArrayList<>();
        long length = 40000;
        long fileSize = 300;
        long outputSize = 300;
        int pesNumber = 1;
        UtilizationModel utilizationModel = new UtilizationModelFull();

        for (int i = 0; i < count; i++) {
            Cloudlet cloudlet = new Cloudlet(i, length, pesNumber, fileSize, outputSize, 
                                           utilizationModel, utilizationModel, utilizationModel);
            cloudlet.setUserId(brokerId);
            cloudlets.add(cloudlet);
        }
        return cloudlets;
    }

    private static void runRoundRobinScheduling() {
        Log.printLine("\n=== Round Robin Scheduling ===");
        resetSimulation();
        
        int vmIndex = 0;
        for (Cloudlet cloudlet : cloudletList) {
            Vm vm = vmList.get(vmIndex % vmList.size());
            cloudlet.setVmId(vm.getId());
            vmIndex++;
        }
        
        CloudSim.startSimulation();
        printResults("Round Robin");
    }

    private static void runGeneticAlgorithmScheduling() {
        Log.printLine("\n=== Genetic Algorithm Scheduling ===");
        resetSimulation();
        
        // GA parameters
        int populationSize = 50;
        int generations = 100;
        double crossoverRate = 0.8;
        double mutationRate = 0.1;
        
        // Initialize population
        List<Map<Cloudlet, Vm>> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            population.add(createRandomSchedule());
        }
        
        // Evolve population
        for (int gen = 0; gen < generations; gen++) {
            // Evaluate fitness
            Map<Map<Cloudlet, Vm>, Double> fitness = new HashMap<>();
            for (Map<Cloudlet, Vm> schedule : population) {
                fitness.put(schedule, calculateMakespan(schedule));
            }
            
            // Selection and reproduction
            List<Map<Cloudlet, Vm>> newPopulation = new ArrayList<>();
            for (int i = 0; i < populationSize; i++) {
                // Tournament selection
                Map<Cloudlet, Vm> parent1 = tournamentSelection(population, fitness, 2);
                Map<Cloudlet, Vm> parent2 = tournamentSelection(population, fitness, 2);
                
                // Crossover
                Map<Cloudlet, Vm> child = crossover(parent1, parent2, crossoverRate);
                
                // Mutation
                mutate(child, mutationRate);
                
                newPopulation.add(child);
            }
            population = newPopulation;
        }
        
        // Find best schedule
        Map<Cloudlet, Vm> bestSchedule = Collections.min(
            population, 
            Comparator.comparingDouble(schedule -> calculateMakespan(schedule))
        );
        
        // Apply best schedule
        for (Cloudlet cloudlet : cloudletList) {
            cloudlet.setVmId(bestSchedule.get(cloudlet).getId());
        }
        
        CloudSim.startSimulation();
        printResults("Genetic Algorithm");
    }

    private static void runACOScheduling() {
        Log.printLine("\n=== ACO Scheduling ===");
        resetSimulation();
        
        int ants = 10;
        int iterations = 50;
        Map<Cloudlet, Vm> bestSchedule = null;
        double bestMakespan = Double.MAX_VALUE;
        
        for (int iter = 0; iter < iterations; iter++) {
            List<Map<Cloudlet, Vm>> antSchedules = new ArrayList<>();
            
            // Each ant constructs a solution
            for (int ant = 0; ant < ants; ant++) {
                Map<Cloudlet, Vm> schedule = new HashMap<>();
                
                for (Cloudlet cloudlet : cloudletList) {
                    // Select VM based on pheromone and heuristic
                    Vm selectedVm = selectVmACO(cloudlet);
                    schedule.put(cloudlet, selectedVm);
                }
                
                antSchedules.add(schedule);
                
                // Evaluate makespan
                double makespan = calculateMakespan(schedule);
                if (makespan < bestMakespan) {
                    bestMakespan = makespan;
                    bestSchedule = schedule;
                }
            }
            
            // Update pheromones
            updatePheromones(antSchedules);
        }
        
        // Apply best schedule
        for (Cloudlet cloudlet : cloudletList) {
            cloudlet.setVmId(bestSchedule.get(cloudlet).getId());
        }
        
        CloudSim.startSimulation();
        printResults("ACO");
    }

    // Helper methods for GA
    private static Map<Cloudlet, Vm> createRandomSchedule() {
        Map<Cloudlet, Vm> schedule = new HashMap<>();
        Random random = new Random();
        
        for (Cloudlet cloudlet : cloudletList) {
            Vm randomVm = vmList.get(random.nextInt(vmList.size()));
            schedule.put(cloudlet, randomVm);
        }
        return schedule;
    }
    
    private static double calculateMakespan(Map<Cloudlet, Vm> schedule) {
        Map<Vm, Double> vmWorkloads = new HashMap<>();
        
        // Initialize workloads for all VMs
        for (Vm vm : vmList) {
            vmWorkloads.put(vm, 0.0);
        }
        
        // Calculate total workload for each VM
        for (Map.Entry<Cloudlet, Vm> entry : schedule.entrySet()) {
            Cloudlet cloudlet = entry.getKey();
            Vm vm = entry.getValue();
            double execTime = estimateExecutionTime(cloudlet, vm);
            vmWorkloads.put(vm, vmWorkloads.get(vm) + execTime);
        }
        
        // Makespan is the maximum workload across all VMs
        return Collections.max(vmWorkloads.values());
    }
    
    private static double estimateExecutionTime(Cloudlet cloudlet, Vm vm) {
        return (cloudlet.getCloudletLength() / (vm.getMips() * vm.getNumberOfPes())) + 
               (cloudlet.getCloudletFileSize() / vm.getBw());
    }
    
    private static Map<Cloudlet, Vm> tournamentSelection(
            List<Map<Cloudlet, Vm>> population, 
            Map<Map<Cloudlet, Vm>, Double> fitness, 
            int tournamentSize) {
        
        Random random = new Random();
        Map<Cloudlet, Vm> best = null;
        double bestFitness = Double.MAX_VALUE;
        
        for (int i = 0; i < tournamentSize; i++) {
            Map<Cloudlet, Vm> candidate = population.get(random.nextInt(population.size()));
            double candidateFitness = fitness.get(candidate);
            
            if (candidateFitness < bestFitness) {
                best = candidate;
                bestFitness = candidateFitness;
            }
        }
        return best;
    }
    
    private static Map<Cloudlet, Vm> crossover(
            Map<Cloudlet, Vm> parent1, 
            Map<Cloudlet, Vm> parent2, 
            double crossoverRate) {
        
        Map<Cloudlet, Vm> child = new HashMap<>();
        Random random = new Random();
        
        for (Cloudlet cloudlet : cloudletList) {
            if (random.nextDouble() < crossoverRate) {
                child.put(cloudlet, parent1.get(cloudlet));
            } else {
                child.put(cloudlet, parent2.get(cloudlet));
            }
        }
        return child;
    }
    
    private static void mutate(Map<Cloudlet, Vm> schedule, double mutationRate) {
        Random random = new Random();
        
        for (Cloudlet cloudlet : cloudletList) {
            if (random.nextDouble() < mutationRate) {
                schedule.put(cloudlet, vmList.get(random.nextInt(vmList.size())));
            }
        }
    }
    
    // Helper methods for ACO
    private static Vm selectVmACO(Cloudlet cloudlet) {
        List<Double> probabilities = new ArrayList<>();
        List<Vm> vms = new ArrayList<>();
        
        for (Vm vm : vmList) {
            double pheromone = pheromoneMatrix.get(cloudlet.getCloudletId()).get(vm.getId());
            double heuristic = 1.0 / estimateExecutionTime(cloudlet, vm);
            double probability = Math.pow(pheromone, ALPHA) * Math.pow(heuristic, BETA);
            
            probabilities.add(probability);
            vms.add(vm);
        }
        
        return selectByProbability(vms, probabilities);
    }
    
    private static Vm selectByProbability(List<Vm> vms, List<Double> probabilities) {
        double sum = probabilities.stream().mapToDouble(Double::doubleValue).sum();
        double threshold = Math.random() * sum;
        double cumulative = 0.0;
        
        for (int i = 0; i < vms.size(); i++) {
            cumulative += probabilities.get(i);
            if (cumulative >= threshold) {
                return vms.get(i);
            }
        }
        return vms.get(vms.size() - 1); // Fallback
    }
    
    private static void updatePheromones(List<Map<Cloudlet, Vm>> antSchedules) {
        // Evaporate pheromones
        for (Map<Integer, Double> vmPheromones : pheromoneMatrix.values()) {
            for (Integer vmId : vmPheromones.keySet()) {
                vmPheromones.put(vmId, vmPheromones.get(vmId) * (1 - RHO));
            }
        }
        
        // Deposit pheromones based on ant solutions
        for (Map<Cloudlet, Vm> schedule : antSchedules) {
            double makespan = calculateMakespan(schedule);
            double pheromoneDeposit = Q / makespan;
            
            for (Map.Entry<Cloudlet, Vm> entry : schedule.entrySet()) {
                Cloudlet cloudlet = entry.getKey();
                Vm vm = entry.getValue();
                
                Map<Integer, Double> vmPheromones = pheromoneMatrix.get(cloudlet.getCloudletId());
                vmPheromones.put(vm.getId(), vmPheromones.get(vm.getId()) + pheromoneDeposit);
            }
        }
    }
    
    // General helper methods
    private static void resetSimulation() {
        // Reset cloudlet assignments
        for (Cloudlet cloudlet : cloudletList) {
            cloudlet.setVmId(-1);
        }
        
        // Clear broker's received lists
        broker.getCloudletReceivedList().clear();
        broker.getCloudletSubmittedList().clear();
        
        // Instead of recreating the broker, just resubmit the lists
        broker.submitGuestList(vmList);
        broker.submitCloudletList(cloudletList);
    }
    
    private static void printResults(String algorithm) {
        List<Cloudlet> finishedCloudlets = broker.getCloudletReceivedList();
        double makespan = 0;
        double totalCost = 0;
        Map<Integer, Double> vmUtilization = new HashMap<>();
        
        // Initialize VM utilization tracking
        for (Vm vm : vmList) {
            vmUtilization.put(vm.getId(), 0.0);
        }
        
        // Calculate metrics
        for (Cloudlet cloudlet : finishedCloudlets) {
            double finishTime = cloudlet.getFinishTime();
            if (finishTime > makespan) {
                makespan = finishTime;
            }
            
            totalCost += cloudlet.getActualCPUTime() * 0.01; // $0.01 per MI
            vmUtilization.put(cloudlet.getVmId(), 
                            vmUtilization.get(cloudlet.getVmId()) + cloudlet.getActualCPUTime());
        }
        
        // Calculate load balance metric
        double avgUtilization = vmUtilization.values().stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
        
        double loadBalanceStdDev = Math.sqrt(
            vmUtilization.values().stream()
                .mapToDouble(util -> Math.pow(util - avgUtilization, 2))
                .average()
                .orElse(0.0)
        );
        
        // Print results
        DecimalFormat df = new DecimalFormat("###.##");
        Log.printLine("\nResults for " + algorithm + ":");
        Log.printLine("Makespan: " + df.format(makespan) + " seconds");
        Log.printLine("Total Cost: $" + df.format(totalCost));
        Log.printLine("Load Balance Std Dev: " + df.format(loadBalanceStdDev));
        Log.printLine("Average VM Utilization: " + df.format(avgUtilization/makespan*100) + "%");
        
        // Print detailed VM utilization
        Log.printLine("\nVM Utilization Details:");
        for (Map.Entry<Integer, Double> entry : vmUtilization.entrySet()) {
            double utilizationPercent = entry.getValue() / makespan * 100;
            Log.printLine("VM " + entry.getKey() + ": " + df.format(utilizationPercent) + "%");
        }
    }
}