import { View } from "react-native"
import MyStyles from "../../styles/MyStyles"
import Categories from "../../components/Categories"

const Home = () => {
    return (
        <View style={MyStyles.container}>
            <Categories />
        </View>
    )
}

export default Home;