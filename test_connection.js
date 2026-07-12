const { db } = require('./database');
const { collection, addDoc, getDocs } = require('firebase/firestore');

async function testConnection() {
  console.log('Testing Firestore database connection for TestSphere...');
  try {
    const testCol = collection(db, '_connection_test_');
    const docRef = await addDoc(testCol, {
      message: 'TestSphere connection successful',
      timestamp: new Date().toISOString()
    });
    console.log('✓ Successfully wrote to database. Document ID:', docRef.id);

    console.log('Reading test documents...');
    const snapshot = await getDocs(testCol);
    console.log(`✓ Successfully read from database. Total docs: ${snapshot.size}`);
    snapshot.forEach(doc => {
      console.log(`- Document [${doc.id}]:`, doc.data());
    });

    console.log('SUCCESS: Connected to Firestore database successfully!');
  } catch (error) {
    console.error('FAILURE: Could not verify connection to Firestore database.');
    console.error('Error Details:', error);
    process.exit(1);
  }
}

testConnection();
