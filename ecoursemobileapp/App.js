import Home from "./screens/Home/Home";
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import Lessons from "./screens/Home/Lessons";
import { NavigationContainer } from "@react-navigation/native";

const Stack = createNativeStackNavigator();
const StackNavigator = () => {
  return (
    <Stack.Navigator>
      <Stack.Screen name="Home" component={Home} options={{ title: "Khóa học" }} />
      <Stack.Screen name="Lesson" component={Lessons} options={{ title: "Bài học" }} />
    </Stack.Navigator>
  )
}

const App = () => {
  return (
    <NavigationContainer>
      <StackNavigator />
    </NavigationContainer>
  )
}

export default App;