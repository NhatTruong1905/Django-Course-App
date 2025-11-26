import { View } from "react-native"
import Categories from "./components/Categories"
import MyStyles from "./styles/MyStyles";
import Home from "./screens/Home/Home";

const App = () => {
  return (
    <View style={MyStyles.container}>
      <Home />
    </View>
  )
};

export default App;