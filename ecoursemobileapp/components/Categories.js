import { useEffect, useState } from "react"
import { View } from "react-native"
import Apis, { endpoints } from "../utils/Apis"
import { Chip } from "react-native-paper"
import MyStyles from "../styles/MyStyles"
import { TouchableOpacity } from "react-native"

const Categories = ({ setCate }) => {
    const [categories, setCategories] = useState([])

    const loadCategories = async () => {
        let res = await Apis.get(endpoints['categories']);
        setCategories(res.data);
    }

    useEffect(() => {
        loadCategories()
    }, [])

    return (
        <View style={MyStyles.row}>
            <TouchableOpacity onPress={() => setCate(null)}>
                <Chip icon='label' style={MyStyles.margin}>Tất cả</Chip>
            </TouchableOpacity>
            {categories.map(c => <TouchableOpacity key={c.id} onPress={() => setCate(c.id)}>
                <Chip icon='label' style={MyStyles.margin}>{c.name}</Chip>
            </TouchableOpacity>)}
        </View>
    );
}

export default Categories;