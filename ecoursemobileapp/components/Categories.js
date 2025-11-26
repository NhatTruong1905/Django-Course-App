import { useEffect, useState } from "react"
import { View } from "react-native"
import Apis, { endpoints } from "../utils/Apis"
import { Chip } from "react-native-paper"
import MyStyles from "../styles/MyStyles"

const Categories = () => {
    const [categories, setCategories] = useState([])

    const loadCategories = async () => {
        let res = await Apis.get(endpoints['categories']);
        setCategories(res.data);
    }

    useEffect(() => {
        loadCategories();
    }, [])

    return (
        <View style={MyStyles.row}>
            {categories.map(c => <Chip icon='label' key={c.id} style={MyStyles.margin}>{c.name}</Chip>)};
        </View>
    );
}

export default Categories;