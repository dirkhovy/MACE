/*
 ***************************************************************************

      USC/ISI MACE Multi-Annotator Competence Estimation  

      USC Information Sciences Institute
      4676 Admiralty Way 
      Marina del Rey, CA 90292-6695 
      USA 

      Original Version: Natural Language Group, April 2013 
      Current Version:  Natural Language Group, April 2013

  Copyright (c) 2013 by the University of Southern California
  All rights reserved.

  Permission to use, copy, modify, and distribute this software and its
  documentation in source and binary forms for any purpose and without
  fee is hereby granted, provided that both the above copyright notice
  and this permission notice appear in all copies, and that any
  documentation, advertising materials, and other materials related to
  such distribution and use acknowledge that the software was developed
  in part by the University of Southern California, Information
  Sciences Institute.  The name of the University may not be used to
  endorse or promote products derived from this software without
  specific prior written permission.

  THE UNIVERSITY OF SOUTHERN CALIFORNIA makes no representations about
  the suitability of this software for any purpose.  THIS SOFTWARE IS
  PROVIDED "AS IS" AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES,
  INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF
  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

  Other copyrights might apply to parts of this software and are so
  noted when applicable.

 ***************************************************************************
 */
package edu.isi;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

/**
 * MACE: Multi-Annotator Competence Estimation
 * 
 * EM-based learning of correct labels and annotator competence
 * 
 * @author tberg, dirkh, avaswani
 *
 */
public class MACE {

    // public static final String VERSION = "0.2"; includes distributions
    // VB as standard
    // TODO: label priors
	public static final String VERSION = "0.3";

	// defaults
	private  static final int DEFAULT_RR = 10;
	private  static final int DEFAULT_ITERATIONS = 50;
	private static final double DEFAULT_NOISE = 0.5;
	private static final double DEFAULT_ALPHA = 0.5;
	private static final double DEFAULT_BETA = 0.5;




	// fields
	public int numInstances;
	public int numAnnotators;
	public int numLabels;


	// training data
	// [d][ai]
	public int[][] whoLabeled;
	// [d][ai]
	public int[][] labels;

	// parameters
	// [a][2]
	public double[][] spamming;
	// [d][l]
	public double[][] thetas;


	// expected counts
	// [d][l]
	public double[][] goldLabelMarginals;
	// [a][l]
	public double[][] strategyExpectedCounts;
	// [a][2]
	public double[][] knowingExpectedCounts;

	// priors
	public double[][] thetaPriors;	// this controls how many of the annotators are actually good

	public double[][] strategyPriors;	// for now, these are 1.0


	double logMarginalLikelhood;
    public double[] entropies;
    
	//hash stuff
	public Map<String, Integer> string2Int;
	public List<String> int2String;
	public int hashCounter;


	/**
	 * Constructor
	 * @param csvFile: comma-separated file, one item per line, each value one annotation
	 * @throws IOException
	 */
	public MACE(String csvFile) throws IOException {

		numInstances = fileLineCount(csvFile);

		labels = new int[numInstances][];
		whoLabeled = new int[numInstances][];
        entropies = new double[numInstances];

		// hash stuff
		string2Int = new HashMap<String, Integer>();
		int2String  = new ArrayList<String>();
		hashCounter = 0;

		// read in CSV file to get all basic information
		this.readFileData(csvFile);

		//this.numAnnotators = annotatorNumber;
		this.numLabels = int2String.size();

		this.goldLabelMarginals = new double[numInstances][numLabels];
		this.strategyExpectedCounts = new double[numAnnotators][numLabels];
		this.knowingExpectedCounts = new double[numAnnotators][2];

	}

	/**
	 * initialize model parameters randomly
	 * @param rand
	 * @param initNoise
	 */
	public void initialize(double initNoise) {
		Random rand = new Random();
		this.spamming = new double[numAnnotators][2];
		this.thetas = new double[numAnnotators][numLabels];
		for (int a=0; a<numAnnotators; ++a) {
			Arrays.fill(spamming[a], 1.0);
			Arrays.fill(thetas[a], 1.0);
			spamming[a][0] += initNoise * rand.nextDouble();
			spamming[a][1] += initNoise * rand.nextDouble();
			for (int l=0; l<numLabels; ++l) {
				thetas[a][l] += initNoise * rand.nextDouble();
			}
		}
		normalizeInPlace(spamming, 0.0);
		normalizeInPlace(thetas, 0.0);

		//System.err.println(thetas[0][0] + " " + thetas[0][1]);
	}

	/**
	 * initialize and set prior matrices
	 * @param initNoise
	 * @param prior
	 */
	public void initialize(double initNoise, double alpha, double beta) {
		this.initialize(initNoise);
		this.thetaPriors = new double[numAnnotators][2];
		this.strategyPriors = new double[numAnnotators][numLabels];
		for (int a=0; a<numAnnotators; ++a) {
			thetaPriors[a][0] = alpha;
			thetaPriors[a][1] = beta;
			Arrays.fill(strategyPriors[a], 10.0);
		}
	}


	/**
	 * compute expected counts
	 */
	public void EStep() {

		// reset counts

		for (int d=0; d<numInstances; ++d) {
			for (int l=0; l<numLabels; ++l) {
				goldLabelMarginals[d][l] = 0.0;
			}
		}
		for (int a=0; a<numAnnotators; ++a) {
			knowingExpectedCounts[a][0] = 0.0;
			knowingExpectedCounts[a][1] = 0.0;
			for (int l=0; l<numLabels; ++l) {
				strategyExpectedCounts[a][l] = 0.0;
			}
		}

		// compute marginals

		logMarginalLikelhood = 0.0;

		for (int d=0; d<numInstances; ++d) {
			double instanceMarginal = 0.0;
			for (int l=0; l<numLabels; ++l) {

                // TODO: add priors here
				double goldLabelMarginal = (1.0 / numLabels);

				for (int ai=0; ai<labels[d].length; ++ai) {
					int a = whoLabeled[d][ai];
					goldLabelMarginal *= spamming[a][0] * thetas[a][labels[d][ai]] + (l == labels[d][ai] ? spamming[a][1] : 0.0);
				}
				instanceMarginal += goldLabelMarginal;
				goldLabelMarginals[d][l] = goldLabelMarginal;
			}

			logMarginalLikelhood += Math.log(instanceMarginal);

			for (int ai=0; ai<labels[d].length; ++ai) {
				int a = whoLabeled[d][ai];
				double strategyMarginal = 0.0;
				for (int l=0; l<numLabels; ++l) {
					strategyMarginal += goldLabelMarginals[d][l] / (spamming[a][0] * thetas[a][labels[d][ai]] + (l == labels[d][ai] ? spamming[a][1] : 0.0));
				}
				strategyMarginal *= spamming[a][0] * thetas[a][labels[d][ai]];
				strategyExpectedCounts[a][labels[d][ai]] += strategyMarginal/instanceMarginal;
				knowingExpectedCounts[a][0] += strategyMarginal/instanceMarginal;
				knowingExpectedCounts[a][1] += (goldLabelMarginals[d][labels[d][ai]] * spamming[a][1] / (spamming[a][0] * thetas[a][labels[d][ai]] + spamming[a][1]))/instanceMarginal;
			}
		}

	}


	/**
	 * compute expected counts when control items are provided
	 */
	public void EStep(Map<Integer,Integer> controls) {

		// reset counts

		for (int d=0; d<numInstances; ++d) {
			for (int l=0; l<numLabels; ++l) {
				goldLabelMarginals[d][l] = 0.0;
			}
		}
		for (int a=0; a<numAnnotators; ++a) {
			knowingExpectedCounts[a][0] = 0.0;
			knowingExpectedCounts[a][1] = 0.0;
			for (int l=0; l<numLabels; ++l) {
				strategyExpectedCounts[a][l] = 0.0;
			}
		}

		// compute marginals

		logMarginalLikelhood = 0.0;

		for (int d=0; d<numInstances; ++d) {
			double instanceMarginal = 0.0;

			for (int l =0; l<numLabels; ++l) {
                // TODO: add priors here
				double goldLabelMarginal = (1.0 / numLabels);

                if (labels[d] != null){
					for (int ai=0; ai<labels[d].length; ++ai) {
						int a = whoLabeled[d][ai];

						goldLabelMarginal *= spamming[a][0] * thetas[a][labels[d][ai]] + (l == labels[d][ai] ? spamming[a][1] : 0.0);
					}

					if (!controls.containsKey(d) || (controls.containsKey(d) && l==controls.get(d))) {
						instanceMarginal +=  goldLabelMarginal;					
						goldLabelMarginals[d][l] = goldLabelMarginal;
					}
				}// if not null
			}


			if (labels[d] != null){
				logMarginalLikelhood += Math.log(instanceMarginal);
				for (int ai=0; ai<labels[d].length; ++ai) {
					int a = whoLabeled[d][ai];
					double strategyMarginal = 0.0;

					if (controls.containsKey(d)){
						if (labels[d][ai]==controls.get(d)) {
							{
								int l = controls.get(d);
								strategyMarginal += goldLabelMarginals[d][l] / (spamming[a][0] * thetas[a][labels[d][ai]] + (l == labels[d][ai] ? spamming[a][1] : 0.0));
							}
							strategyMarginal *= spamming[a][0] * thetas[a][labels[d][ai]];
							strategyExpectedCounts[a][labels[d][ai]] += strategyMarginal/instanceMarginal;
							knowingExpectedCounts[a][0] += strategyMarginal/instanceMarginal;
							knowingExpectedCounts[a][1] += (goldLabelMarginals[d][labels[d][ai]] * spamming[a][1] / (spamming[a][0] * thetas[a][labels[d][ai]] + spamming[a][1]))/instanceMarginal;
						} else {
							strategyExpectedCounts[a][labels[d][ai]] += 1.0;
							knowingExpectedCounts[a][0] += 1.0;
						}
					} else{
						for (int l=0; l<numLabels; ++l) {
							strategyMarginal += goldLabelMarginals[d][l] / (spamming[a][0] * thetas[a][labels[d][ai]] + (l == labels[d][ai] ? spamming[a][1] : 0.0));
						}
						strategyMarginal *= spamming[a][0] * thetas[a][labels[d][ai]];
						strategyExpectedCounts[a][labels[d][ai]] += strategyMarginal/instanceMarginal;
						knowingExpectedCounts[a][0] += strategyMarginal/instanceMarginal;
						knowingExpectedCounts[a][1] += (goldLabelMarginals[d][labels[d][ai]] * spamming[a][1] / (spamming[a][0] * thetas[a][labels[d][ai]] + spamming[a][1]))/instanceMarginal;
					}
				}//for
			}// if not null
		}

	}


	/**
	 * normalize expected counts
	 * @param smoothing
	 */
	public void MStep(double smoothing) {
		spamming = normalize(knowingExpectedCounts, smoothing);
		thetas = normalize(strategyExpectedCounts, smoothing);
	}

	/**
	 * normalize using priors
	 */
	public void variationalMStep() {
		spamming = variationalNormalize(knowingExpectedCounts, thetaPriors);
		thetas = variationalNormalize(strategyExpectedCounts, strategyPriors);
	}


	/**
	 * find best answer under the current model, ignore instance above threshold
	 * @return answer vector
	 */
	public String[] decode(double threshold) {
		// get entropies
		entropies = getLabelEntropies();
		double entropyThreshold = getEntropyForThreshold(threshold, Arrays.copyOf(entropies, numInstances));

		String[] result = new String[numInstances];
		for (int d=0; d<numInstances; ++d) {
			double bestProb = Double.NEGATIVE_INFINITY;
			int bestLabel = -1;

//			System.err.println(entropies[d]);
//			System.err.println(entropyThreshold);
			// ignore instances above threshold
			if (entropies[d] <= entropyThreshold){

				// empty lines
				if (entropies[d] == Double.NEGATIVE_INFINITY){
					result[d] = "";					
				} 

				// otherwise, find best entropy
				else{
					for (int l=0; l<numLabels; ++l) {

						if (goldLabelMarginals[d][l] > bestProb) {
							bestProb = goldLabelMarginals[d][l];
							bestLabel = l;
						}
					}
					result[d] = int2String.get(bestLabel);
				}//else
			}
			else
				result[d] = "";
		}

		return result;
	}



	public String[] decodeDistribution(double threshold) {
		// get entropies
		entropies = getLabelEntropies();
		double entropyThreshold = getEntropyForThreshold(threshold, Arrays.copyOf(entropies, numInstances));

		String[] result = new String[numInstances];
		for (int d=0; d<numInstances; ++d) {

			// ignore instances above threshold 
			if (entropies[d] <= entropyThreshold){

				// empty lines
				if (entropies[d] == Double.NEGATIVE_INFINITY){
					result[d] = "";					
				} 

				// otherwise, find best entropy
				else{

					Map<String, Double> distribution = new HashMap<String, Double>();
                    double distributionTotal = 0.0;
					for (int l=0; l<numLabels; ++l) {
						distribution.put(int2String.get(l), goldLabelMarginals[d][l]);
                        distributionTotal += goldLabelMarginals[d][l];
					}

					distribution = sortByValue(distribution);
					String[] output = new String[numLabels];
					int position = numLabels-1;
					for(Map.Entry<String, Double> entry : distribution.entrySet()) {
						output[position] = entry.getKey() + " " + entry.getValue()/distributionTotal;
						position -= 1;
					}

					StringBuffer outputString = new StringBuffer(); 
					for (int l=0; l<numLabels; ++l) {
						outputString.append(output[l]);
						if (l < numLabels-1){
							outputString.append("\t");						
						} 
					}
					result[d] = outputString.toString();
				}
			}
			else
				result[d] = "";
		}

		return result;
	}


	/**
	 * helper function for decodeDistribution(), to sort the labels by their entropy
	 * from http://stackoverflow.com/questions/109383/how-to-sort-a-mapkey-value-on-the-values-in-java
	 * @param map
	 * @return
	 */
	public static <K, V extends Comparable<? super V>> Map<K, V> 
	sortByValue( Map<K, V> map )
	{
		List<Map.Entry<K, V>> list =
				new LinkedList<Map.Entry<K, V>>( map.entrySet() );
		Collections.sort( list, new Comparator<Map.Entry<K, V>>()
				{
			public int compare( Map.Entry<K, V> o1, Map.Entry<K, V> o2 )
			{
				return (o1.getValue()).compareTo( o2.getValue() );
			}
				} );

		Map<K, V> result = new LinkedHashMap<K, V>();
		for (Map.Entry<K, V> entry : list)
		{
			result.put( entry.getKey(), entry.getValue() );
		}
		return result;
	}






	/**
	 * 
	 * @return the entropies of each instance
	 */
	public double[] getLabelEntropies(){
		double[] result = new double[numInstances];

		for (int d=0; d<numInstances; ++d) {
			if (labels[d] == null){
				result[d] = Double.NEGATIVE_INFINITY;
			}
			else{
				double norm = 0.0;
				double entropy = 0.0;
				for (int l=0; l<numLabels; ++l) {
					norm += goldLabelMarginals[d][l];
				}
				for (int l=0; l<numLabels; ++l) {
					double p = goldLabelMarginals[d][l]/norm;
					if (p > 0.0) {
						entropy += -p * Math.log(p);
					}
				}
				result[d] = entropy;
			}//else
		}	

		return result;
	}



	/**
	 * run EM with the specified parameters
	 * @param beta 
	 * @param controls: filename of controls, one item per line 
	 * @param numIters: number of iterations
	 * @param smoothing: smoothing added to expected counts before normalizing
	 * @param numRestarts: number of restarts
	 * @throws IOException 
	 */
	public void run(int numIters, double smoothing, int numRestarts, double alpha, double beta, boolean variational, String controlsFile) throws IOException {
		Map<Integer, Integer> controls;
		if (controlsFile != null){
			controls = this.readControls(controlsFile);
		}
		else{
			controls = new HashMap<Integer, Integer>();
		}

		double[][] bestThetas = new double[numAnnotators][2];
		double[][] bestStrategies = new double[numAnnotators][numLabels];
		double bestLogMarginalLikelihood = Double.NEGATIVE_INFINITY;
		int rrBestModelOccurredAt = 0;

        System.err.print("Running ");
        if (variational){
            System.err.print("Variational Bayes");
        }
        else{
            System.err.print("vanilla");
        }
        
		System.err.println(" EM training with the following settings:");
		System.err.println("\t"+numIters+" iterations");
		System.err.println("\t"+numRestarts+" restarts");
		System.err.println("\tsmoothing = "+smoothing);
		if (variational){
			System.err.println("\talpha = "+alpha);
			System.err.println("\tbeta = "+beta);			
		}

		double start = System.currentTimeMillis();
		for (int rr=0; rr<numRestarts; rr++){
			System.err.println("\n============");
			System.err.println("Restart " + (rr+1) );
			System.err.println("============");

			// initialize
			if (variational)
				initialize(DEFAULT_NOISE, alpha, beta);
			else
				initialize(DEFAULT_NOISE);

			// run first E-Step to get counts
			EStep(controls);
			System.err.println("initial log marginal likelihood = "+logMarginalLikelhood);

			// iterate
			for (int t=0; t<numIters; ++t) {
				if (variational)
					variationalMStep();
				else
					MStep(smoothing);
				EStep(controls);
				//System.err.println("iter "+t);
				//System.err.println("log marginal likelihood "+logMarginalLikelhood);
			}
			System.err.println("final log marginal likelihood = "+logMarginalLikelhood);

			// renormalize thetas
			//normalizeInPlace(thetas, 0.0);
			//normalizeInPlace(strategies, 0.0);

			if (logMarginalLikelhood > bestLogMarginalLikelihood){
				//if (rr>0) System.err.println("NEW BEST MODEL!\n");
				rrBestModelOccurredAt = rr+1;
				bestLogMarginalLikelihood = logMarginalLikelhood;
				bestThetas = spamming.clone();
				bestStrategies = thetas.clone();
			}
		}
		System.err.println("\nTraining completed in " + ((System.currentTimeMillis() - start) / 1000) + "sec");
		System.err.println("Best model came from random restart number " + rrBestModelOccurredAt + " (log marginal likelihood: " + bestLogMarginalLikelihood + ")");
		logMarginalLikelhood = bestLogMarginalLikelihood;
		spamming = bestThetas;
		thetas = bestStrategies;

		// run E-step to get marginals of latest model
		EStep(controls);

	}

	/**
	 * normalize a matrix by row
	 * @param mat
	 * @param smoothing
	 * @return normalized matrix
	 */
	public static double[][] normalize(double[][] mat, double smoothing) {
		double[][] result = new double[mat.length][mat[0].length];
		for (int i=0; i<result.length; ++i) {
			double norm = 0.0;
			for (int j=0; j<result[0].length; ++j) {
				norm += mat[i][j] + smoothing;
			}
			for (int j=0; j<result[0].length; ++j) {
				if (norm > 0.0) result[i][j] = (mat[i][j] + smoothing) / norm;
			}
		}
		return result;
	}

	/**
	 * normalize a matrix by row using hyperparameters 
	 * @param mat
	 * @param hyperparameters: a matrix with the priors
	 * @return normalized matrix
	 */
	public static double[][] variationalNormalize(double[][] mat, double[][] hyperparameters) {
		double[][] result = new double[mat.length][mat[0].length];
		for (int i=0; i<result.length; ++i) {
			double norm = 0.0;
			for (int j=0; j<result[0].length; ++j) {
				norm += mat[i][j] + hyperparameters[i][j];
			}
			norm = Math.exp(edu.illinois.dais.ttr.Mathematics.digamma(norm));
			for (int j=0; j<result[0].length; ++j) {
				if (norm > 0.0) {
					result[i][j] = Math.exp(edu.illinois.dais.ttr.Mathematics.digamma( (mat[i][j] + hyperparameters[i][j]) )) / norm;
				}
			}
		}
		return result;
	}


	/**
	 * normalize a matrix by row, in place
	 * @param mat
	 * @param smoothing
	 */
	public static void normalizeInPlace(double[][] mat, double smoothing) {
		for (int i=0; i<mat.length; ++i) {
			double norm = 0.0;
			for (int j=0; j<mat[0].length; ++j) {
				norm += mat[i][j] + smoothing;
			}
			for (int j=0; j<mat[0].length; ++j) {
				if (norm > 0.0) mat[i][j] = (mat[i][j] + smoothing) / norm;
			}
		}
	}


	/**
	 * read in a file with control items
	 * @param fileName
	 * @return
	 * @throws IOException
	 */
	public Map<Integer, Integer> readControls(String fileName) throws IOException {
		Map<Integer, Integer> controls = new HashMap<Integer, Integer>();

		String line = null;
		try {
			//FileReader fr1 = new FileReader(fileName);
			InputStream inputStream = new FileInputStream(fileName);
			Reader      reader      = new InputStreamReader(inputStream, "UTF-8");
			//BufferedReader br1 = new BufferedReader(fr1);
			BufferedReader br1 = new BufferedReader(reader);

			System.err.println("Reading controls file " + fileName);

			int lineNumber = 0;
			while ((line = br1.readLine()) != null) {
				line = line.trim();

				// record item
				if (!line.equals("")){

					// record value if not hashed yet
					if (!string2Int.containsKey(line)){
						string2Int.put(line, hashCounter++);
						int2String.add(line);
					}

					controls.put(lineNumber, string2Int.get(line) );
				}

				lineNumber++;
			}
			br1.close();
			return controls;

		}catch(IOException e){
			throw new IOException(e.getMessage());
		}

		catch (ArrayIndexOutOfBoundsException aiobe){
			throw new IOException("Problem reading file: array out of bounds. Please check whether the file contains at least one line, and that it is formatted in UTF-8.");
		}


	}


	/**
	 * read CSV file and record data
	 * @param fileName
	 * @throws IOException
	 */
	public void readFileData(String fileName) throws IOException {
		String line = null;
		try {
			//FileReader fr1 = new FileReader(fileName);
			InputStream inputStream = new FileInputStream(fileName);
			Reader      reader      = new InputStreamReader(inputStream, "UTF-8");
			//BufferedReader br1 = new BufferedReader(fr1);
			BufferedReader br1 = new BufferedReader(reader);

			System.err.println("Reading CSV file '" + fileName + "'");

			int lineNumber = 0;
			while ((line = br1.readLine()) != null) {
				if (lineNumber>0){
					if (lineNumber%5==0) System.err.print(".");
					if (lineNumber%100==0) System.err.println(lineNumber);
				}

				// handle empty lines
				if (line.trim().equals("")){
					// store as int[][]
					labels[lineNumber] = null;
					whoLabeled[lineNumber] = null;
				}


				//regular lines
				else{
					List<Integer> annotatorsOnItem = new ArrayList<Integer>();
					List<Integer> itemValues = new ArrayList<Integer>();

					// split into items
					StringBuilder token = new StringBuilder("");
					int readPosition = 0;
					int annotatorNumber = 0;
					char[] charArray = line.toCharArray();
					for (char c: charArray){
						// record values
						if (c != ',')
							token.append(c);

						// record after a comma or at the end of the line
						if (c == ',' || readPosition+1==charArray.length){

							// last item can be comma or value, so check for both before you break
							String item = token.toString();

							// reset token
							token = new StringBuilder("");

							// record item
							if (!item.equals("")){

								// record which annotator gave an answer
								annotatorsOnItem.add(annotatorNumber);

								// record value
								if (!string2Int.containsKey(item)){
									string2Int.put(item, hashCounter++);
									int2String.add(item);
								}
								itemValues.add( string2Int.get(item) );
							}

						}


						// advance reading position
						readPosition++;

						// separate annotators
						if (c == ',')
							annotatorNumber++;

					}

					if (numAnnotators > 0 && annotatorNumber+1 != this.numAnnotators){
						throw new IOException("number of annotations in line " + (lineNumber+1) + " differs from previous line!");
					}
					this.numAnnotators = annotatorNumber+1;

					// store as int[][]
					labels[lineNumber] = toIntArray(itemValues);
					whoLabeled[lineNumber] = toIntArray(annotatorsOnItem);
				}// regular lines


				lineNumber++;
			}// while there are lines left

			br1.close();

			System.err.println("\nstats:\n\t" + lineNumber + " instances,\n\t" + int2String.size() + " labels "+int2String+",\n\t" + numAnnotators + " annotators\n");

		}// try

		catch (IOException e) {
			// catch possible io errors from readLine()
			throw new IOException(e.getMessage());
		}// catch

		catch (ArrayIndexOutOfBoundsException aiobe){
			throw new IOException("Problem reading file: array out of bounds. Please check whether the file contains at least one line, and that it is formatted in UTF-8.");
		}


	}


	/**
	 * print the arrays
	 * @param someArray
	 */
	public void printIntArray(int[][] someArray){
		for (int i = 0; i < someArray.length; i++){
			printIntArray(someArray[i]);
		}
	}
	/**
	 * print the arrays
	 * @param someArray
	 */
	public void printIntArray(int[] someArray){
		for (int i = 0; i < someArray.length; i++){
			System.out.print(String.valueOf(someArray[i]) + " ");
		}
		System.out.println();
	}


	/**
	 * turn an ArrayList into a primitive array
	 * @param list
	 * @return int[]
	 */
	public int[] toIntArray(List<Integer> list){
		int[] ret = new int[list.size()];
		for(int i = 0;i < ret.length;i++)
			ret[i] = list.get(i);
		return ret;
	}

	/**
	 * count the number of lines in a file (to initialize arrays)
	 * @param filename
	 * @return
	 * @throws IOException
	 */
	public int fileLineCount(String filename) throws IOException {
		InputStream inStream = new BufferedInputStream(new FileInputStream(filename));
		try {
			byte[] character = new byte[1024];
			int lineCount = 0;
			int readChars = 0;
			while ((readChars = inStream.read(character)) != -1) {
				for (int i = 0; i < readChars; ++i) {
					if (character[i] == '\n')
						++lineCount;
				}
			}
			return lineCount;
		} finally {
			inStream.close();
		}
	}


	/**
	 * write a double[] array to a file, using a delimiter between elements (usually either tab or newline) 
	 * @param array
	 * @param fileName
	 * @param delimiter
	 * @throws IOException 
	 */
	public void writeArrayToFile(Object[] array, String fileName, String delimiter) throws IOException{
		PrintWriter pr = null;    
		try
		{
			System.err.print("writing to file '" + fileName + "'...");
			pr = new PrintWriter(fileName);
			int i=0;
			for (; array.length>0 && i<array.length-1;i++){
				pr.print(array[i]);
				pr.print(delimiter);
			}

			pr.println(array[i]);
		}
		catch (Exception e)
		{
			throw new IOException(e);
		}
		finally{
			pr.close();			
			System.err.print("done\n");
		}

	}

	/**
	 * sort entropies, get value corresponding to n
	 * @return the entropy value at n% of the entropy values
	 */
	public double getEntropyForThreshold(double threshold, double[] entropyArray){
		double result = Double.NEGATIVE_INFINITY;
		int pivot;

		if (threshold == 0.0)
			pivot = 0;
		else if (threshold == 1.0)
			pivot = numInstances-1;
		else
			pivot = (int) (numInstances*threshold);

        //entropyArray = getLabelEntropies();
        //entropyArray = entropies.copy();
		Arrays.sort(entropyArray);
		result = entropyArray[pivot];

		return result;
	}


	/**
	 * evaluate the current model by comparing to a test file with gold labels
	 * @param testFile
	 * @param threshold 
	 * @return the accuracy
	 * @throws IOException 
	 */
	@SuppressWarnings("finally")
	public double test(String testFile, String[] predictions) throws IOException{
		double accuracy = 0.0;
		double correct = 0.0;
		double total = 0.0;

		int numLinesInTest = fileLineCount(testFile);
		if (numLinesInTest != numInstances)
			throw new IOException("Number of lines in test file does not match!");

		String line = null;
		try {
			FileReader fr1 = new FileReader(testFile);
			BufferedReader br1 = new BufferedReader(fr1);

			System.err.println("Reading test file");

			int lineNumber = 0;
			while ((line = br1.readLine()) != null) {
				// only consider instances that were below the threshold
				if (!predictions[lineNumber].equals("")){
					line = line.trim();

					total++;
					if (line.equals(predictions[lineNumber]))
						correct++;
				}
				lineNumber++;
			}
			br1.close();
			System.out.print("Coverage: " + total/numInstances + "\t");
			accuracy = correct/total;

		}catch(IOException e){
			throw new IOException(e.getMessage());
		}
		finally{
			return accuracy;			
		}

	}






	//==========================================================================
	private static void doc(){
		System.err.println("MACE -- Multi-Annotator Confidence Estimation");
		System.err.println("============================================");
        System.err.println("Authors: Taylor Berg-Kirkpatrick, Dirk Hovy, Ashish Vaswani");
		MACE.getVersion();

        System.err.println("Usage:\t MACE [options] CSV_FILE\n");
        System.err.println("Note: MACE runs Variational Bayes EM training by default. If you would like to use vanilla EM, set the --em parameter");
		System.err.println("Options:");
		System.err.println("=========");
		System.err.println("\t--controls <FILE>:\tsupply a file with annotated control items. Each line corresponds to one item,\n" +
				"\t\t\t\tso the number of lines MUST match the input CSV file.\n" +
				"\t\t\t\tThe control items serve as semi-supervised input. Controls usually improve accuracy.\n");
		System.err.println("\t--alpha <FLOAT>:\tfirst hyper-parameter of beta prior that controls whether an annotator knows or guesses. Default:" + MACE.DEFAULT_ALPHA + "\n");
		System.err.println("\t--beta <FLOAT>:\t\tsecond hyper-parameter of beta prior that controls whether an annotator knows or guesses. Default:" + MACE.DEFAULT_BETA + "\n");
		System.err.println("\t--distribution:\t\tfor each items, list all labels and their entropy in '[prefix.]prediction'\n");
        System.err.println("\t--entropies:\t\twrite the entropy of each instance to a separate file '[prefix.]entropy'\n");
        System.err.println("\t--em:\t\t\tuse EM training rather than Variational Bayes training\n");
		System.err.println("\t--help:\t\t\tdisplay this information\n");
		System.err.println("\t--iterations <1-1000>:\tnumber of iterations for each EM start. Default: " + MACE.DEFAULT_ITERATIONS + "\n");
		System.err.println("\t--prefix <STRING>:\tprefix used for output files.\n");
		System.err.println("\t--restarts <1-1000>:\tnumber of random restarts to perform. Default: " + MACE.DEFAULT_RR + "\n");
		System.err.println("\t--smoothing <0.0-1.0>:\tsmoothing added to fractional counts before normalization.\n" +
				"\t\t\t\tHigher values mean smaller changes. Default: 0.01/|values|\n");
		System.err.println("\t--test <FILE>:\t\tsupply a test file. Each line corresponds to one item in the CSV file,\n" +
				"\t\t\t\tso the number of lines must match. If a test file is supplied,\n" +
				"\t\t\t\tMACE outputs the accuracy of the predictions\n");
		System.err.println("\t--threshold <0.0-1.0>:\tonly predict the label for instances whose entropy is among the top n%, ignore others.\n" +
				"\t\t\t\tThus '--threshold 0.0' will ignore all instances, '--threshold 1.0' includes all.\n" +
				"\t\t\t\tThis improves accuracy at the expense of coverage. Default: 1.0\n");

		System.err.println();
		System.err.println("To cite MACE in publications, please refer to:");
		System.err.println("Dirk Hovy, Taylor Berg-Kirkpatrick, Ashish Vaswani and Eduard Hovy (2013): Learning Whom to Trust With MACE. In: Proceedings");

		System.err.println();
		System.err.println("This is research software that is not actively maintained. If you have any questions, please write to <dirkh@isi.edu>");

		System.exit(0);
	}

	private static void getVersion() {
		System.err.println("Version: " + MACE.VERSION + "\n");		
	}

	public static void main(String[] args) {
		MACE em = null;
		try {
			if (args.length == 0 || args[0].equals("--help")){
				MACE.doc();
			}
			if (args[0].equals("--version")){
				MACE.getVersion();
				System.exit(0);
			}


			int numberOfArgs = args.length;
			String file = args[numberOfArgs-1];
			em = new MACE(file);

			// default settings
			int iterations = MACE.DEFAULT_ITERATIONS;
			int restarts = MACE.DEFAULT_RR;
			double smoothing = 0.01 / (double)em.numLabels;
			double threshold = 1.0;
			String test = null;
			String controls = null;
			String prefix = null;
			boolean entropies = false;
			boolean distribution = false;
			boolean variational = true;
			double alpha = MACE.DEFAULT_ALPHA;
			double beta = MACE.DEFAULT_BETA;

			// process all but last arg (which is the CSV file)
			for (int i = 0; i < numberOfArgs-1; i++){
				String arg = args[i];

				if (arg.equals("--smoothing")){
					smoothing = Double.valueOf(args[++i]);
					if (smoothing < 0.0)
						throw new IllegalArgumentException("smoothing less than 0.0");
				}

				else if (arg.equals("--threshold")){
					threshold = Double.valueOf(args[++i]);
					if (threshold < 0.0 || threshold > 1.0)
						throw new IllegalArgumentException("threshold not between 0.0 and 1.0");
				}

				else if (arg.equals("--restarts")){
					restarts = Integer.valueOf(args[++i]);
					if (restarts < 1 || restarts > 1000)
						throw new IllegalArgumentException("restarts not between 1 and 1000");
				}

				else if (arg.equals("--iterations")){
					iterations = Integer.valueOf(args[++i]);
					if (iterations  < 1 || iterations > 1000)
						throw new IllegalArgumentException("iterations not between 1 and 1000");
				}

				else if (arg.equals("--prefix")){
					prefix = args[++i];
				}

				else if (arg.equals("--entropies")){
					entropies = true;
				}

                else if (arg.equals("--em")){
                    variational = false;
                }

                else if (arg.equals("--distribution")){
					distribution = true;
				}

				else if (arg.equals("--controls")){
					controls = args[++i];
				}

				else if (arg.equals("--test")){
					test = args[++i];
				}

				else if (arg.equals("--help")){
					MACE.doc();
				}

				else if (arg.equals("--version")){
					MACE.getVersion();
				}

				else if (arg.equals("--alpha")){
					alpha = Double.valueOf(args[++i]);
					variational = true;
				}
				else if (arg.equals("--beta")){
					beta = Double.valueOf(args[++i]);
					variational = true;
				}

				else
					throw new IllegalArgumentException("argument '"+arg +"' not recognized");
			}


			// run with configuration
			em.run(iterations, smoothing, restarts, alpha, beta, variational, controls);

			// write results to files
			// generate predictions
			String[] predictions;
			if (!distribution){
				predictions = em.decode(threshold);
			}
			else {
				predictions = em.decodeDistribution(threshold); 
			}
			String predictionName = prefix == null ? "prediction" : prefix + ".prediction";
			em.writeArrayToFile(predictions, predictionName, "\n");

			// generate competence scores
			Object[] competence = new Object[em.numAnnotators];
			for (int i=0; i < em.numAnnotators; i++){
				competence[i] = em.spamming[i][1];
			}
			String competenceName = prefix == null ? "competence" : prefix + ".competence";
			em.writeArrayToFile(competence, competenceName, "\t");

			// generate entropies
			if (entropies){
				Object[] entropy = new Object[em.numInstances];
//				double[] entropyArray = em.getLabelEntropies();
                double[] entropyArray = em.entropies;
				for (int i=0; i < em.numInstances; i++)
                    entropy[i] = entropyArray[i] == Double.NEGATIVE_INFINITY ? "" : entropyArray[i];
				String entropyName = prefix == null ? "entropies" : prefix + ".entropies";
				em.writeArrayToFile(entropy, entropyName, "\n");
			}

			if (test != null){
				System.out.println("Accuracy on test set: " + em.test(test, predictions));
			}

		}catch (IOException e) {
			System.err.println("\n*****************************************************************");
			System.err.println("\tFILE ERROR:");
			System.err.println("\t" + e.getMessage());
			System.err.println("*****************************************************************");
		}catch (IllegalArgumentException e) {
			System.err.println("\n*****************************************************************");
			System.err.println("\tARGUMENT ERROR:");
			System.err.println("\t" + e.getMessage());
			System.err.println("*****************************************************************");
			MACE.doc();
		}
	}
}


