import { View } from "react-native"
import MyStyles from "../../styles/MyStyles"
import Categories from "../../components/Categories"
import Courses from "../../components/Courses"
import { useState } from "react"

const Home = () => {
    const [cate, setCate] = useState();

    return (
        <View style={MyStyles.container}>
            <Categories setCate={setCate} />
            <Courses cate={cate} />
        </View>
    )
}

export default Home;    