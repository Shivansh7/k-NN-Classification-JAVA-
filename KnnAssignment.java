package ucd.aml.assignment;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Scanner;

/**
 * 
 * @author shivansh AML Assignment Knn Implementation
 * 
 */
public class KnnAssignment {

	public static String[] docNames; // The document names
	public static String[] allUniqueWords; // The unique words in the documents
	public static String[] wordString; // The String array with all the words of
										// a document
	public static String[] wordFrequencyString; // The String array with all the
												// frequencies of words
	public static String[] allClassLabels; // The string array with all the
											// classlabels
	public static ArrayList<InputDocumentClass> allDocuments = new ArrayList<InputDocumentClass>(),
			testingDataSet = new ArrayList<>(), trainingDataSet = new ArrayList<InputDocumentClass>();
	public static int totalNoOfInputDocuments = 0;
	public static float testSetPercentage = 15f; // the test size %age
	public static ArrayList<String> words = new ArrayList<>(), freq = new ArrayList<>();
	public static int kVal = 0; // the K value for KNN
	public static String mtxLoc, labelLoc; // the data file locations

	public static void main(String[] args) {

		BufferedReader bufferedReaderMtx = null, bufferedReaderLabel = null;
		int docCount = 0;
		String content = new String();
		try {
			takeInput(); // take the user inputs for file locations and K value
			bufferedReaderMtx = new BufferedReader(new FileReader(new File(mtxLoc)));
			bufferedReaderLabel = new BufferedReader(new FileReader(new File(labelLoc)));

			int skipHeadersCount = 0;
			InputDocumentClass obj = new InputDocumentClass();
			while ((content = bufferedReaderMtx.readLine()) != null) {
				String[] lineRead = content.split(" ");
				if (content != null && skipHeadersCount > 1) {
					if (!searchValue(docNames, lineRead[0])) {
						allDocuments.add(obj);
						docNames[docCount] = lineRead[0];
						String labelContent = bufferedReaderLabel.readLine();
						String[] splitLabel = labelContent.split(",");
						String document = splitLabel[0];
						String label = splitLabel[1];
						allClassLabels[Integer.parseInt(document) - 1] = label;
						obj = new InputDocumentClass();
						obj.documentStringName = lineRead[0]; // document name
																// from the
																// readline
						docCount++;
					}

				} else if (skipHeadersCount == 1) {
					initializeArrayValues(lineRead); // the header file
														// containing the size
														// of all th rows , to
														// initialize all the
														// arrays
				}

				if (skipHeadersCount > 1) {
					obj.wordList.add(lineRead[1]);
					obj.wordFrequencyList.add(lineRead[2]);
				}
				skipHeadersCount++;

			}
			totalNoOfInputDocuments = docCount;

			// calculate the training and testing documents from the user
			// specified inputs
			trainingDataSet = new ArrayList<InputDocumentClass>(
					allDocuments.subList(0, (int) (((100f - testSetPercentage) / 100) * totalNoOfInputDocuments)));
			testingDataSet = new ArrayList<InputDocumentClass>(allDocuments.subList(trainingDataSet.size(),
					trainingDataSet.size() + (int) ((testSetPercentage / 100) * totalNoOfInputDocuments)));
			System.out.println(" >> Training Set Size : " + trainingDataSet.size() + " << "
					+ "\n >> Testing Set Size : " + testingDataSet.size() + " << ");

			// pass the test and training data to compute cosine similarities
			computeSimilarity();

		} catch (IOException ioe) {
			ioe.printStackTrace();
		} finally {
			try {
				if (bufferedReaderMtx != null)
					bufferedReaderMtx.close();
				if (bufferedReaderLabel != null)
					bufferedReaderLabel.close();
			} catch (IOException ioe) {
				ioe.getMessage();
			}
		}

	}

	// asks the user for various input values
	private static void takeInput() {
		// TODO Auto-generated method stub
		Scanner inputSc = new Scanner(System.in);
		System.out.println("\n\n*********************** AML Assignment : KNN ***************************\n");
		System.out.println(" Enter the file locations : \n Input location for data file .mtx :");
		mtxLoc = inputSc.nextLine();
		System.out.println(" Input location for class labels file .labels :");
		labelLoc = inputSc.nextLine();
		System.out.println("\nEnter the K Value - ");
		kVal = inputSc.nextInt();
		System.out.print("\nEnter the testing size percentage to train :");
		testSetPercentage = (inputSc.nextInt());

	}

	// Computes the cosine similarity scores for all the test documents with all
	// the
	// training sets
	private static void computeSimilarity() {
		// TODO Auto-generated method stub
		ArrayList<CosineSimilarityClass> allCosineVals = new ArrayList<>();
		ArrayList<CosineSimilarityClass> tsCosineVals = new ArrayList<>();

		System.out.println("\nExecuting the algorithm...Wait for sometime, until the program executes...\n");
		trainingDataSet.remove(0);
		for (InputDocumentClass tsData : testingDataSet) {
			int i = 0;
			for (InputDocumentClass trData : trainingDataSet) {
				ArrayList<String> commonWords = new ArrayList<>(tsData.wordList);
				commonWords.retainAll(trData.wordList); // retain the common
														// words from both docs
				int dotProductSum = 0;
				for (String common : commonWords) { // cosine similarity ,
													// calculates the dot
													// products
					dotProductSum += (Integer
							.parseInt(trData.wordFrequencyList.get(getIndexFromArraylists(trData.wordList, common))))
							* (Integer.parseInt(
									tsData.wordFrequencyList.get(getIndexFromArraylists(tsData.wordList, common))));
				}

				float freqProductSum1 = getSquaredSumsOfFrequencies(tsData.wordFrequencyList);
				float freqProductSum2 = getSquaredSumsOfFrequencies(trData.wordFrequencyList);
				float cosineSimValue = dotProductSum / (freqProductSum1 * freqProductSum2);
				tsCosineVals.add(new CosineSimilarityClass(tsData.documentStringName + ":" + trData.documentStringName,
						cosineSimValue));
			}
			// sort the cosine similarity list in increasing order and then
			// reverse it to get the descending order
			Collections.sort(tsCosineVals, new Comparator<CosineSimilarityClass>() {
				@Override
				public int compare(CosineSimilarityClass o1, CosineSimilarityClass o2) {
					return Float.compare(o1.value, o2.value);
				}
			});
			Collections.reverse(tsCosineVals);

			// get the top K elements and pass it to the classifier
			tsCosineVals = new ArrayList<>(tsCosineVals.subList(0, kVal));
			allCosineVals.addAll(tsCosineVals);
			tsCosineVals = new ArrayList<CosineSimilarityClass>();
			i++;
		}
		generateClassifierAccuracy(allCosineVals);
	}

	// It calculates the classifier accuracies for both the weighted and
	// unweighted classifiers
	// Uses the cosine similarities as calculated and predicts the class labels
	// on sorted values
	// to predict the accuracies.
	private static void generateClassifierAccuracy(ArrayList<CosineSimilarityClass> allCosineVals) {
		float correctU = 0f, correctW = 0f;
		for (int i = 0; i < allCosineVals.size();) {

			String testDocName = "";
			ArrayList<String> trainingDocClassLabel = new ArrayList<>();
			ArrayList<String> trainingDocClassWeights = new ArrayList<>();
			ArrayList<String> allUniqueClasslabels = new ArrayList<>();
			ArrayList<Float> allUniqueClasslabelsWts = new ArrayList<>();

			// runs for the specified K values and forms the adequate clusters
			for (int j = 0; j < kVal; j++) {
				CosineSimilarityClass obj = allCosineVals.get(i + j);
				String[] nameVal = obj.documentString.split(":");
				testDocName = nameVal[0];
				String trainingDocName = nameVal[1];
				float cosineVal = obj.value;
				float weightVal = (float) Math.pow(1 - cosineVal, -1);
				String lab = allClassLabels[Integer.parseInt(trainingDocName)];
				trainingDocClassLabel.add(lab);
				if (getIndexFromArraylists(allUniqueClasslabels, lab) < 0)
					allUniqueClasslabels.add(lab);
				trainingDocClassWeights.add(String.valueOf(weightVal) + ":" + lab);
			}

			// calculates the maximum occuring or the majority class label from
			// the K nearest neighbours
			int maxFreq = 0, max = 0;
			for (int k = 0; k < trainingDocClassLabel.size(); k++) {
				int freq = Collections.frequency(trainingDocClassLabel, trainingDocClassLabel.get(k));
				if (maxFreq < freq) {
					maxFreq = freq;
					max = k;
				}
			}

			// checks the predicted value with the actual class label
			String predctionofClass = trainingDocClassLabel.get(max);
			String originalClassDocLable = allClassLabels[Integer.parseInt(testDocName)];
			if (predctionofClass.equals(originalClassDocLable))
				correctU++;

			// adds all the sum of weights for the distinct vectors finds out
			// the highest weighted class
			Float[] sumWts = new Float[kVal];
			Float maxWt = 0f;
			int maxW = 0;
			for (int m = 0; m < allUniqueClasslabels.size(); m++) {
				String labl = allUniqueClasslabels.get(m);
				sumWts[m] = 0f;
				for (int n = 0; n < trainingDocClassWeights.size(); n++) {
					if (trainingDocClassWeights.get(n).contains(labl))
						sumWts[m] += Float.parseFloat(trainingDocClassWeights.get(n).split(":")[0]);
				}
				if (sumWts[m] > maxWt) {
					maxWt = sumWts[m];
					maxW = m;
				}
			}

			String predctionofClassWtd = allUniqueClasslabels.get(maxW);
			if (predctionofClassWtd.equals(originalClassDocLable))
				correctW++;

			i = i + kVal;

		}

		// print the calculated accuracies to the console
		float accrU = (correctU / testingDataSet.size()) * 100;
		float accrW = (correctW / testingDataSet.size()) * 100;
		System.out
				.println("\n >> Unweighted Accuracy: " + accrU + "% <<" + "\n >> Weighted Accuracy: " + accrW + "% <<");
	}

	// returns the vector values for each document, by taking the square root of
	// the sum of squares of the freq
	private static float getSquaredSumsOfFrequencies(ArrayList<String> wordFrequencyList) {
		// TODO Auto-generated method stub
		float sum = 0f;
		for (int i = 0; i < wordFrequencyList.size(); i++) {
			sum += Math.pow(Float.parseFloat(wordFrequencyList.get(i)), 2);
		}
		return (float) Math.sqrt(sum);
	}

	// To initialize all the arrays with default " " values and prevent null
	// pointers
	private static void initializeArrayValues(String[] lineRead) {
		// TODO Auto-generated method stub
		int totalDocs = Integer.parseInt(lineRead[0]);
		docNames = new String[totalDocs];
		allClassLabels = new String[totalDocs];
		for (int i = 0; i < totalDocs; i++) {
			docNames[i] = "";
			allClassLabels[i] = "";
		}

	}

	// Returns the index of the target string from the array, else -1
	public static int getIndexFromArraylists(ArrayList<String> list, String targetString) {
		int i = 0;
		for (String key : list) {
			if (key.equals(targetString)) {
				return i;
			} else {
				i++;
				continue;
			}
		}
		return -1;

	}

	// This functions returns the index of the specified string from the
	// arraylist, else -1 if not found
	public static boolean searchValue(String[] arr, String targetValue) {
		for (String s : arr) {
			if (s.equals(targetValue))
				return true;
		}
		return false;
	}

	// The meta data class containing all the input document data variables
	public static class InputDocumentClass {

		public String documentStringName; // the document name
		public ArrayList<String> wordList; // its word list
		public ArrayList<String> wordFrequencyList; // the frequency of words

		public InputDocumentClass() {
			wordList = new ArrayList<>();
			wordFrequencyList = new ArrayList<>();
		}

	}

	// The meta data class with all the document class and their cosine
	// similarity values
	public static class CosineSimilarityClass {

		public String documentString;
		public float value;

		public CosineSimilarityClass() {

		}

		// The meta data clas with all the cosine values
		public CosineSimilarityClass(String documentStringName, float cosineSineValue) {
			// TODO Auto-generated constructor stub
			this.documentString = documentStringName;
			this.value = cosineSineValue;
		}

	}
}
