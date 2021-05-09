package score

import (
	"fmt"
	"github.com/garyburd/redigo/redis"
	"strconv"
)

//func Score(state *framework.CycleState, node *nodeinfo.NodeInfo) (int64, error) {
//	s, err := CalculateCollectScore(state, node)
//	if err != nil {
//		return 0, err
//	}
//	s += CalculatePodUseScore(node)
//	return s, nil
//}

func Score(nodeName string) (int64, error) {
	conn, err := redis.Dial("tcp",
		"192.168.2.104:23234",
		redis.DialDatabase(0), //DialOption参数可以配置选择数据库、连接密码等
		redis.DialPassword("root123456"))
	if err != nil {
		//fmt.Println("Connect to redis failed ,cause by >>>", err)
		return 0, nil
	}
	defer conn.Close()

	//如果db有密码，可以设置
	//if _,err := conn.Do("AUTH","password");err !=nil{
	//	fmt.Println("connect db by pwd failed >>>",err)
	//}

	//写入值{"test-Key":"test-Value"}
	//_, err = conn.Do("SET", "test-Key", "test-Value", "EX", "5")
	//if err != nil {
	//	fmt.Println("redis set value failed >>>", err)
	//}
	//
	//time.Sleep(10 * time.Second)
	//检查是否存在key值
	//exists, err := redis.Bool(conn.Do("EXISTS", "name"))
	//if err != nil {
	//	fmt.Println("illegal exception")
	//}
	//fmt.Printf("exists or not: %v \n", exists)

	//read value
	v, err := redis.String(conn.Do("GET", nodeName))
	if err != nil {
		fmt.Println("redis get value failed >>>", err)
	}
	//fmt.Println("get value: ", v)

	res, err := strconv.ParseInt(v, 10, 64)
	if err != nil {
		res = 1
	}

	return res, nil
}
