from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(['tokyo', 'osaka', 'nagoya', 'tokyo', 'yokohama', 'osaka'])
import pdb; pdb.set_trace()
le.transform(['tokyo', 'osaka', 'nagoya', 'tokyo', 'yokohama', 'osaka'])
print()
