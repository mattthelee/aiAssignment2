// ECS629/759 Assignment 2 - ID3 Skeleton Code
// Author: Simon Dixon

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;
//import Double;

class ID3 {

	/** Each node of the tree contains either the attribute number (for non-leaf
	 *  nodes) or class number (for leaf nodes) in <b>value</b>, and an array of
	 *  tree nodes in <b>children</b> containing each of the children of the
	 *  node (for non-leaf nodes).
	 *  The attribute number corresponds to the column number in the training
	 *  and test files. The children are ordered in the same order as the
	 *  Strings in strings[][]. E.g., if value == 3, then the array of
	 *  children correspond to the branches for attribute 3 (named data[0][3]):
	 *      children[0] is the branch for attribute 3 == strings[3][0]
	 *      children[1] is the branch for attribute 3 == strings[3][1]
	 *      children[2] is the branch for attribute 3 == strings[3][2]
	 *      etc.
	 *  The class number (leaf nodes) also corresponds to the order of classes
	 *  in strings[][]. For example, a leaf with value == 3 corresponds
	 *  to the class label strings[attributes-1][3].
	 **/
	class TreeNode {

		TreeNode[] children;
		int value;

		public TreeNode(TreeNode[] ch, int val) {
			value = val;
			children = ch;
		} // constructor

		public String toString() {
			return toString("");
		} // toString()
		
		String toString(String indent) {
			if (children != null) {
				String s = "";
				for (int i = 0; i < children.length; i++)
					s += indent + data[0][value] + "=" +
							strings[value][i] + "\n" +
							children[i].toString(indent + '\t');
				return s;
			} else
				// Here is is indexing the last attribute because that is in fact the class
				return indent + "Class: " + strings[attributes-1][value] + "\n";
		} // toString(String)

	} // inner class TreeNode

	private int attributes; 	// Number of attributes (including the class)
	private int examples;		// Number of training examples
	private TreeNode decisionTree;	// Tree learnt in training, used for classifying
	private String[][] data;	// Training data indexed by example, attribute
	private String[][] strings; // Unique strings for each attribute
	private int[] stringCount;  // Number of unique strings for each attribute

	public ID3() {
		attributes = 0;
		examples = 0;
		decisionTree = null;
		data = null;
		strings = null;
		stringCount = null;
	} // constructor
	
	public void printTree() {
		if (decisionTree == null)
			error("Attempted to print null Tree");
		else
			System.out.println(decisionTree);
	} // printTree()

	/** Print error message and exit. **/
	static void error(String msg) {
		System.err.println("Error: " + msg);
		System.exit(1);
	} // error()

	static final double LOG2 = Math.log(2.0);
	
	static double xlogx(double x) {
		return x == 0? 0: x * Math.log(x) / LOG2;
	} // xlogx()

	/** Execute the decision tree on the given examples in testData, and print
	 *  the resulting class names, one to a line, for each example in testData.
	 **/
	public void classify(String[][] testData) {
		if (decisionTree == null)
			error("Please run training phase before classification");
		// PUT  YOUR CODE HERE FOR CLASSIFICATION
		String [][] trimmedData = Arrays.copyOfRange(testData, 1, testData.length);

		for (String[] testInstance : trimmedData){
			System.out.println(strings[attributes-1][predictClass(testInstance, decisionTree)]);
		}
        // continue until every data point is in a leafnode
        // print the data and associated classes

	} // classify()

	public void train(String[][] trainingData) {
		indexStrings(trainingData);
		// Get rid of first line of the array which has variable names in
		String [][] trimmedData = Arrays.copyOfRange(trainingData, 1, trainingData.length);
		List <Integer> attributeList = getAttributeList();

		// Performs some test to sanity check the functions
		System.out.println("Beginning tests");
		System.out.println("This should be -0.442 : " + xlogx(0.6));
		System.out.println("Entropy calculation of training data (0.94 for realestate):" + entropy(trimmedData));
		System.out.println("Modeclass should return most prevalent class string (yes on realestate): " + strings[attributes-1][getModeClass(trainingData)]);
		System.out.println("Best attribute index to split on (should be for 0 realestate): " + getBestSplitAttIndex(trimmedData,attributeList));
		System.out.println("End of tests");

		decisionTree = dtLearn(trimmedData, attributeList);
		//System.out.println("\n*****Final Tree*****");
		//printTree();

	} // train()

	public int predictClass (String[] instance, TreeNode node){
		// Returns the class for the given data instance (row)

		// If we have reached a lefnode return the classification
		if (node.children == null){
			// classify points with decisionTree.value
			return node.value;
		}
		//System.out.println("Debug: " + node.children.length);
		for (int i = 0; i< node.children.length; i++){
			// If this instance's attribute value matches the childs index
			//System.out.println(instance[node.value]);
			//System.out.println(strings[node.value][i]);

			//System.out.println(instance[node.value].equals(strings[node.value][i]));
			if (instance[node.value].equals(strings[node.value][i])){
				// Continue expansion to next child
				return predictClass(instance, node.children[i]);
			}
		}
		error("Something has gone wrong in the predictClass function");
		return -1;
	}

	public TreeNode dtLearn(String [][] data, List<Integer> attributeList){
		// Trains the decision tree,
		// based off of decision tree learning algorithm in Russell and Norvig Chapter 18, pg702
		if (data.length == 0){
			error("No data passed to this dtLearn");
		} else if (attributeList.size() == 0 || entropy(data) == 0){
			// If there are no more attributes or data is all one class then return leafnode
			int classIndex = getModeClass(data);
			return new TreeNode(null, classIndex);
		}

		int bestSplitAttributeIndex = getBestSplitAttIndex(data, attributeList);
		//System.out.println("Debug " + bestSplitAttributeIndex);
		TreeNode tree = new TreeNode(new TreeNode[stringCount[bestSplitAttributeIndex]],bestSplitAttributeIndex);
		for (int j =0; j < stringCount[bestSplitAttributeIndex]; j++){
			// Get the split of the data possible on this attribute
			String[][] dataSplit = splitData(data,bestSplitAttributeIndex,j);
			List<Integer> newAttributeList = new ArrayList(attributeList);
			newAttributeList.remove(new Integer(bestSplitAttributeIndex));
			TreeNode subtree = dtLearn(dataSplit, newAttributeList);
			//System.out.println("Debug2: " + subtree);

			tree.children[j] = subtree;
		}
		//System.out.println("Returning tree:" );
		//System.out.println(tree);

		return tree;
	}

	public int getBestSplitAttIndex(String[][] data, List<Integer> attributeList){
		// Returns the best attribute by finding the splits with lowest total entropy
		// Try each possible split
		double minEnt = Double.POSITIVE_INFINITY;
		int bestSplitAttributeIndex = 0;
		for ( int attributeIndex : attributeList){
			double entropy = 0;

			// For each node, split them into their subnodes based on the attribute selected and the possible values for the attribute
			for (int j =0; j < stringCount[attributeIndex]; j++){
				String [][] dataSplit = splitData(data,attributeIndex,j);
				entropy += dataSplit.length*entropy(dataSplit)/ data.length;
			}
			// Something is wrong withthe calculation here as i need to multiply by the number of values in that bucket
			//System.out.println("Entropy for attributeSplit " + attributeIndex + " is: " + entropy);
			if (entropy < minEnt){
				minEnt = entropy;
				bestSplitAttributeIndex = attributeIndex;
			}
		}
		return bestSplitAttributeIndex;
	}

	public List<Integer> getAttributeList(){
		// Returns a list of all the attribute indexes possible
		List<Integer> attributeList = new ArrayList<Integer>();
		// -1 is to skip the class as a splitabble attribute
		for (int i =0 ; i < attributes -1; i++){
			attributeList.add(i);
		}
		return attributeList;
	}

	public int getModeClass(String[][] data){
		// Returns the index of the class that is most prevalent
		// Ties are broken by choosing lower index class
		int [] classCounts = new int[stringCount[attributes-1]];
		// For each class, count the number of instances of that class
		for (int i =0 ; i < stringCount[attributes-1]; i++){
			for (int j = 0; j < data.length; j++){
				if (data[j][attributes-1].equals( strings[attributes-1][i])){
					classCounts[i]++;
				}
			}
		}
		// TODO could make this more efficent by putting in the loop above
		// Get the string of the highest count class
		int maxCount = 0;
		int modeClass = 0;
		for (int k =0 ; k < classCounts.length; k++){
			int count = classCounts[k];
			if (count > maxCount){
				maxCount = count;
				modeClass = k;
			}
		}
		return modeClass;
	}

    public double entropy(String[][] data){
		// Gives the entropy of one split of the data
		double entropy = 0;
		//System.out.println("DEBUG: " + stringCount[attributes-1]);

		// For each class that exists in dataset
		for (int i =0; i< stringCount[attributes-1]; i++){
	        // Count the number of datapoints with that class
			int count = 0;
			for (int j = 0; j < data.length; j++){
				//System.out.println("Count " + j + " are thees equal: " + data[j][attributes-1] + ":" + strings[attributes-1][i]);
				//System.out.println(data[j][attributes-1].equals(strings[attributes-1][i]));
				// If datapoint's class matches current class
				if (data[j][attributes-1].equals( strings[attributes-1][i])){
					count++;
				}
			}

			// Do the log formula thing for this cass and add it to the entropy
			entropy += -xlogx((float) count / data.length);
			//System.out.println("Count of class" + i + " : " + count + " datalength: " + data.length + " entropy: " + -xlogx((float)count / data.length));
        }
		return entropy;
    }

    public String[][] splitData(String[][] data, int attributeIndex, int valueIndex){
	    // Returns the subset of data that has the same value for the given attribute
        String value = strings[attributeIndex][valueIndex];
	    List<Integer> splitIndexes = new ArrayList<Integer>();
	    for (int rowIndex =0; rowIndex < data.length; rowIndex++){
	        if (data[rowIndex][attributeIndex].equals(strings[attributeIndex][valueIndex])){
				splitIndexes.add(rowIndex);
            }
        }
		String[][] dataSplit = new String[splitIndexes.size()][attributes];
	    for ( int i = 0; i < splitIndexes.size(); i++){
	    	dataSplit[i] = data[splitIndexes.get(i)];
		}

		return dataSplit;
    }

	/** Given a 2-dimensional array containing the training data, numbers each
	 *  unique value that each attribute has, and stores these Strings in
	 *  instance variables; for example, for attribute 2, its first value
	 *  would be stored in strings[2][0], its second value in strings[2][1],
	 *  and so on; and the number of different values in stringCount[2].
	 **/
	void indexStrings(String[][] inputData) {
        data = inputData;
		examples = data.length;
		attributes = data[0].length;
		stringCount = new int[attributes];
		strings = new String[attributes][examples];// might not need all columns
		int index = 0;
		for (int attr = 0; attr < attributes; attr++) {
			stringCount[attr] = 0;
			for (int ex = 1; ex < examples; ex++) {
				for (index = 0; index < stringCount[attr]; index++)
					if (data[ex][attr].equals(strings[attr][index]))
						break;	// we've seen this String before
				if (index == stringCount[attr])		// if new String found
					strings[attr][stringCount[attr]++] = data[ex][attr];
			} // for each example
		} // for each attribute
	} // indexStrings()

	/** For debugging: prints the list of attribute values for each attribute
	 *  and their index values.
	 **/
	void printStrings() {
		for (int attr = 0; attr < attributes; attr++)
			for (int index = 0; index < stringCount[attr]; index++)
				System.out.println(data[0][attr] + " value " + index +
									" = " + strings[attr][index]);
	} // printStrings()
		
	/** Reads a text file containing a fixed number of comma-separated values
	 *  on each line, and returns a two dimensional array of these values,
	 *  indexed by line number and position in line.
	 **/
	static String[][] parseCSV(String fileName)
								throws FileNotFoundException, IOException {
		BufferedReader br = new BufferedReader(new FileReader(fileName));
		String s = br.readLine();
		int fields = 1;
		int index = 0;
		while ((index = s.indexOf(',', index) + 1) > 0)
			fields++;
		int lines = 1;
		while (br.readLine() != null)
			lines++;
		br.close();
		String[][] data = new String[lines][fields];
		Scanner sc = new Scanner(new File(fileName));
		sc.useDelimiter("[,\n]");
		for (int l = 0; l < lines; l++)
			for (int f = 0; f < fields; f++)
				if (sc.hasNext())
					data[l][f] = sc.next();
				else
					error("Scan error in " + fileName + " at " + l + ":" + f);
		sc.close();
		return data;
	} // parseCSV()

	public static void main(String[] args) throws FileNotFoundException,
												  IOException {
		if (args.length != 2)
			error("Expected 2 arguments: file names of training and test data");
		String[][] trainingData = parseCSV(args[0]);
		String[][] testData = parseCSV(args[1]);
		ID3 classifier = new ID3();
		classifier.train(trainingData);
		classifier.printTree();
		classifier.classify(testData);
	} // main()

} // class ID3

