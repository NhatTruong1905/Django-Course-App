import { useEffect, useState } from "react";
import Apis, { endpoints } from "../../utils/Apis";
import { ActivityIndicator, FlatList } from "react-native";
import { List } from "react-native-paper";
import MyStyles from "../../styles/MyStyles";

const Lesson = ({ route }) => {
    const [lessons, setLessons] = useState([]);
    const [loading, setLoading] = useState(false);
    const courseId = route.params?.courseId

    const loadLessons = async () => {
        try {
            setLoading(true)

            let res = await Apis.get(endpoints['lessons'](courseId))
            setLessons(res.data)
        } catch (ex) {
            console.error(ex)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        loadLessons()
    }, [courseId])

    return (
        <FlatList ListFooterComponent={loading && <ActivityIndicator size="large" />}
            data={lessons} renderItem={({ item }) => <List.Item
                title={item.subject}
                description={item.created_date}
                left={() => <Image source={{ uri: item.image }} style={MyStyles.avatar} />}
            />} />
    )
}

export default Lesson 