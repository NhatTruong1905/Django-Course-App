import { useEffect, useState } from "react"
import { ActivityIndicator, FlatList, Image } from "react-native";
import MyStyles from "../styles/MyStyles";
import { List, Searchbar } from "react-native-paper";
import Apis, { endpoints } from "../utils/Apis";
import { TouchableOpacity } from "react-native";

const Courses = ({ cate, navigation }) => {
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(false);
    const [q, setQ] = useState("");
    const [page, setPage] = useState(1);

    const loadCourses = async () => {
        try {
            setLoading(true)

            let url = `${endpoints['courses']}?page=${page}`

            if (q) {
                url = `${url}&q=${q}`
            }

            if (cate) {
                url = `${url}&category_id=${cate}`
            }

            console.info(url)

            let res = await Apis.get(url)

            if (res.data.next === null) {
                setPage(0)
            }

            if (page === 1) {
                setCourses(res.data.results)
            }
            else if (page > 1)
                setCourses([...courses, ...res.data.results])

        } catch (ex) {
            console.error(ex)
        } finally {
            setLoading(false)
        }
    }

    useEffect(() => {
        let timer = setTimeout(() => {
            if (page > 0)
                loadCourses();
        }, 500)

        return () => clearTimeout(timer)
    }, [q, page, cate])

    useEffect(() => {
        setPage(1)
    }, [q, cate])

    const loadMore = () => {
        if (page > 0 && !loading)
            setPage(page + 1)
    }

    return (
        <>
            <Searchbar placeholder="Tìm khóa học..." value={q} onChangeText={setQ} />
            <FlatList ListFooterComponent={loading && <ActivityIndicator size="large" />}
                onEndReached={loadMore} data={courses} renderItem={({ item }) => <List.Item
                    title={item.subject}
                    description={item.created_date}
                    left={() => <TouchableOpacity onPress={() => navigation.navigate("Lesson", { "courseId": item.id })} >
                        <Image source={{ uri: item.image }} style={MyStyles.avatar} />
                    </TouchableOpacity>}
                />} />
        </>
    );
}

export default Courses;