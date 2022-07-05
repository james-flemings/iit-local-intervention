import iit
import model
from sklearn.metrics import classification_report

def main():
  lower, upper = 0, 9 
  embed_dim = 5 
  size = abs(lower) + abs(upper)
  num_addends = 4 
  output_size = num_addends * size + 1
  num_layers = 2
  #output_size = 1

  addends_train, sums_train = iit.generate_data(10000, num_addends, 0,
                                  lower, upper)  

  addends_test, sums_test = iit.generate_data(250, num_addends, 1)  

  mod = model.TorchDeepNeuralClassifier(size + 1, output_size, num_addends,
                                    num_layers, embed_dim, max_iter=250)

  mod.fit(addends_train, sums_train)
  preds = mod.predict(addends_train)

  print("\nClassification report:")
  print(classification_report(sums_train, preds))

if __name__ == '__main__':
  main()