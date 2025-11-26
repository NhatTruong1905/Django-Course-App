import { useState } from "react"
import { FlatList } from "react-native";

const Courses = () => {
    const [courses, setCourses] = useState([]);
    const [loading, setLoading] = useState(false);

    const loadCourses = async () => {
        try {
            setCourses(true)
        } catch (ex) {
            console.error(ex)
        } finally {
            setCourses(false)
        }
    }

    return (
        <>
            <FlatList />
        </>
    )
}