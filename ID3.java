// ECS629/759 Assignment 2 - ID3 Skeleton Code
// Author: Simon Dixon

import java.io.File;
import java.io.FileReader;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Scanner;
import Double;

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

        // For the dataset, starting at the root, split the data based on the criteria
        // continue until every data point is in a leafnode
        // print the data and associated classes

	} // classify()

	public void train(String[][] trainingData) {
		indexStrings(trainingData);
		// PUT  YOUR CODE HERE FOR TRAINING
		System.out.println("Beginning tests");
		System.out.println("This should be -0.736966 : " + xlogx(0.6));
		System.out.println("Entropy calculation from first 3 rows of training data:" + entropy(trainingData,[0,1,2]));
		System.out.println("End of tests");

        // Try each possible split
        double minEnt = Double.POSITIVE_INFINITY;
        for ( int attributeIndex = 0; attributeIndex < attributes.length; attributeIndex++){
            double entropy = 0;

            // For each node, split them into their subnodes based on the attribute selected and the possible values for the attribute
            for (int j =0; j < stringCount[attributeIndex]; j++){
                List<Integer> dataSplit = splitData(trainingData,trainingIndices,attributeIndex,j);
                entropy += entropy(trainingData,datasplit);
            }
            if (entropy < minEnt){
                minEnt = entropy;
                int bestSplitAttributeIndex = attributeIndex;
            }
        }
        currentNode.value = bestSplitAttributeIndex;
        int childNo = stringCount[bestSplitAttributeIndex];
        currentNode.children = new TreeNode[childNo];
        for (int i; i <childNo; i++ ){
        	currentNode.children = new TreeNode(null,null);
		}


        // Get the split with the maximum entropy and set this as the decision tree
        // Repeat for each node until running out of attributes to classify or entropy is 0 for the node - recursion
        // Save final structure of tree
	} // train()

    public double entropy(String[][] data,  List<Integer> dataIndexes){
		// Gives the entropy of one split of the data
        // For each class that exists in dataset
		double entropy = 0;
        for (int i; i< stringCount[-1]; i++){
	        // Count the number of datapoints with that class
			int count = 0;
			for (int j; j < dataIndexes.length; j++){
				// If datapoint's class matches current class
				if (data[dataIndexes[j]][-1] == stringCount[-1][i]){
					count++;
				}
			}
			// Do the log formula thing for this cass and add it to the entropy
			entropy += - (count/dataIndexes.length)*xlogx(count / dataIndexes.length);
        }


		return entropy;
    }

    public List<Integer> splitData(String[][] data, List<Integer> dataIndexes, int attributeIndex, int valueIndex){
	    // Returns the subset of data that has the same value for the given attribute
        String value = strings[attributeIndex][valueIndex];
	    List<Integer> dataSplit = new ArrayList<Integer>();
	    for (int rowIndex : dataIndexes){
	        if (data[rowIndex][attributeIndex] = strings[attributeIndex][valueIndex]){
	            dataSplit.add(rowIndex);
            }
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
        List<Integer> trainingIndices;
        data = inputData;
		examples = data.length;
		attributes = data[0].length;
		stringCount = new int[attributes];
		strings = new String[attributes][examples];// might not need all columns
		int index = 0;
		for (int attr = 0; attr < attributes; attr++) {
			stringCount[attr] = 0;
			for (int ex = 1; ex < examples; ex++) {
			    if (attr == 0){
                    trainingIndices.add(ex -1)
                }
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